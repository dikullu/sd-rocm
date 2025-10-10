#!/usr/bin/env python3
import atexit
import datetime as _dt
import os
import sys
import threading
import time
import traceback
import runpy
import signal
import gzip

# Attempt to import torch lazily to avoid hard failure if cpuonly
try:
    import torch
    from torch.profiler import profile, ProfilerActivity
except Exception as e:
    torch = None
    profile = None
    ProfilerActivity = None


def _now_str():
    return _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _read_rss_kb_fallback():
    """Read RSS (resident set size) in KB using /proc/self/statm to avoid extra deps."""
    try:
        with open('/proc/self/statm', 'r') as f:
            parts = f.read().strip().split()
        if not parts:
            return None
        pages = int(parts[1])  # resident pages
        page_size = os.sysconf('SC_PAGE_SIZE')  # bytes per page
        return (pages * page_size) // 1024
    except Exception:
        return None


def _write_text_atomic(path: str, content: str):
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class MemorySampler:
    def __init__(self, out_path: str, interval: float = 2.0, use_cuda: bool = True):
        self.out_path = out_path
        self.interval = interval
        self.use_cuda = use_cuda
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='gpu_mem_sampler', daemon=True)

    def start(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, 'a', encoding='utf-8') as f:
            f.write(f"# Memory sampler started at {_now_str()}\n")
            f.write("timestamp, rss_kb, cuda_mem_allocated_mb, cuda_mem_reserved_mb, cuda_max_allocated_mb, cuda_max_reserved_mb\n")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            with open(self.out_path, 'a', encoding='utf-8') as f:
                f.write(f"# Memory sampler stopped at {_now_str()}\n")
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                rss_kb = _read_rss_kb_fallback()
                if self.use_cuda and torch is not None and torch.cuda.is_available():
                    try:
                        dev = torch.cuda.current_device()
                    except Exception:
                        dev = 0
                    try:
                        allocated = torch.cuda.memory_allocated(dev)
                        reserved = torch.cuda.memory_reserved(dev)
                        max_allocated = torch.cuda.max_memory_allocated(dev)
                        max_reserved = torch.cuda.max_memory_reserved(dev)
                    except Exception:
                        allocated = reserved = max_allocated = max_reserved = 0
                else:
                    allocated = reserved = max_allocated = max_reserved = 0

                row = f"{_now_str()}, {rss_kb or ''}, {allocated/1024/1024:.3f}, {reserved/1024/1024:.3f}, {max_allocated/1024/1024:.3f}, {max_reserved/1024/1024:.3f}\n"
                with open(self.out_path, 'a', encoding='utf-8') as f:
                    f.write(row)
            except Exception:
                # Don't crash the app if sampler fails
                try:
                    with open(self.out_path, 'a', encoding='utf-8') as f:
                        f.write(f"# Sampler error at {_now_str()}:\n{traceback.format_exc()}\n")
                except Exception:
                    pass
            finally:
                time.sleep(self.interval)


def main():
    # Expected to be invoked as: python profiler_runner.py <comfyui_main_and_args...>
    if len(sys.argv) < 2:
        print("Usage: profiler_runner.py <script.py> [args...]", file=sys.stderr)
        sys.exit(2)

    # Env configuration
    profile_dir = os.environ.get('PROFILE_DIR')
    if not profile_dir:
        # Default under current working directory
        ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        profile_dir = os.path.abspath(os.path.join(os.getcwd(), f"profiles/{ts}"))
    os.makedirs(profile_dir, exist_ok=True)

    sampling_interval = float(os.environ.get('PROFILING_SAMPLING_INTERVAL', '2.0'))
    enable_torch_prof = os.environ.get('PROFILING_TORCH_PROFILER', '1') not in ('0', 'false', 'False', '')
    compress_exports = os.environ.get('PROFILING_EXPORT_GZIP', '1') not in ('0', 'false', 'False', '')

    # Start memory sampler
    sampler = MemorySampler(os.path.join(profile_dir, 'memory_log.csv'), interval=sampling_interval)
    sampler.start()

    # Start torch profiler (if torch present)
    p_holder = {'p': None, 'seq': 0}
    p_lock = threading.Lock()

    def _start_profiler_locked():
        # call only with p_lock held
        if not (enable_torch_prof and torch is not None):
            p_holder['p'] = None
            return
        try:
            acts = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            # Read size-related toggles from env to limit trace size
            record_shapes_flag = os.environ.get('PROFILING_RECORD_SHAPES', '0') not in ('0', 'false', 'False', '')
            with_stack_flag = os.environ.get('PROFILING_WITH_STACK', '0') not in ('0', 'false', 'False', '')
            p = profile(
                activities=acts,
                record_shapes=record_shapes_flag,
                profile_memory=True,
                with_stack=with_stack_flag,
            )
            p.__enter__()
            p_holder['p'] = p
        except Exception:
            p_holder['p'] = None

    _start_profiler_locked()

    shutdown_called = {'v': False}

    def _maybe_gzip(src_path: str):
        try:
            gz_path = src_path + '.gz'
            with open(src_path, 'rb') as fin, gzip.open(gz_path, 'wb') as fout:
                # Stream copy to avoid high memory usage
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            # fsync compressed file to disk
            try:
                with open(gz_path, 'rb') as fh:
                    os.fsync(fh.fileno())
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _export_snapshot_locked(path: str) -> bool:
        # call only with p_lock held
        p = p_holder['p']
        if p is None:
            return False
        tmp_path = path + '.tmp'
        try:
            p.export_chrome_trace(tmp_path)
            with open(tmp_path, 'rb') as fh:
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
            if compress_exports:
                _maybe_gzip(path)
            return True
        except Exception as ex:
            # Log error for diagnostics
            errlog = os.path.join(profile_dir, 'profiler_error.log')
            try:
                _write_text_atomic(errlog, f"[{_now_str()}] snapshot export failed:\n{traceback.format_exc()}\n")
            except Exception:
                pass
            print(f"[profiler] Failed to export trace: {ex}", file=sys.stderr)
            return False

    def _finalize_export():
        with p_lock:
            p = p_holder['p']
            if p is None:
                return False
            trace_path = os.path.join(profile_dir, 'torch_trace.json')
            ok = _export_snapshot_locked(trace_path)
            # After final export, close/stop profiler
            try:
                p.__exit__(None, None, None)
            except Exception:
                pass
            try:
                if hasattr(p, 'stop'):
                    p.stop()
            except Exception:
                pass
            p_holder['p'] = None
            return ok

    def _shutdown(signum=None, frame=None):
        if shutdown_called['v']:
            return
        shutdown_called['v'] = True
        try:
            _finalize_export()
        finally:
            try:
                sampler.stop()
            except Exception:
                pass
        if signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    # Register handlers
    atexit.register(_shutdown)
    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        # Optional on-demand export without stopping the server
        def _export_only(signum, frame):
            try:
                with p_lock:
                    # Rotate the profiler to ensure finalized buffers before exporting
                    p = p_holder['p']
                    if p is None:
                        # Nothing to export; start a new profiler if profiling is enabled
                        _start_profiler_locked()
                        print("[profiler] Profiler not active or export failed.", file=sys.stderr)
                        return

                    # Close current profiler first to flush data
                    try:
                        p.__exit__(None, None, None)
                    except Exception:
                        pass
                    try:
                        if hasattr(p, 'stop'):
                            p.stop()
                    except Exception:
                        pass

                    # Export to timestamped file to avoid conflicts
                    p_holder['seq'] += 1
                    ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
                    trace_path = os.path.join(profile_dir, f'torch_trace-{ts}-{p_holder["seq"]}.json')
                    ok = False
                    try:
                        # Now that profiler is finalized, export should succeed
                        # Temporarily assign into holder for export helper (expects p_holder['p'])
                        p_holder_backup = p_holder['p']
                        p_holder['p'] = p
                        ok = _export_snapshot_locked(trace_path)
                    finally:
                        # Clear holder since this profiler instance is closed
                        p_holder['p'] = None

                    # Immediately start a new profiler for continued recording
                    _start_profiler_locked()

                if ok:
                    print("[profiler] On-demand trace exported.")
                else:
                    print("[profiler] Profiler not active or export failed.", file=sys.stderr)
            except Exception:
                pass
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, _export_only)
    except Exception:
        # Some platforms may not allow installing handlers (e.g., Windows in some cases)
        pass

    # Forward execution to the target script (ComfyUI's main.py) with its args
    target_script = sys.argv[1]
    target_args = sys.argv[1:]
    sys.argv = target_args

    # Resolve target script to absolute path and ensure its directory is the CWD and on sys.path
    try:
        if not os.path.isabs(target_script):
            target_script = os.path.abspath(os.path.join(os.getcwd(), target_script))
        target_dir = os.path.dirname(target_script)
        # Prepend target_dir to sys.path so package imports (e.g., 'comfy') work
        if target_dir and target_dir not in sys.path:
            sys.path.insert(0, target_dir)
        # Change working directory to the target script directory
        os.chdir(target_dir or os.getcwd())
    except Exception:
        # If anything goes wrong, continue; run_path below may still work
        pass

    # Make sure Python prints unbuffered when running server
    try:
        os.environ.setdefault('PYTHONUNBUFFERED', '1')
    except Exception:
        pass

    # Run the script as if executed directly
    try:
        runpy.run_path(target_script, run_name='__main__')
    finally:
        # Ensure cleanup even if target raises
        _shutdown()


if __name__ == '__main__':
    main()
