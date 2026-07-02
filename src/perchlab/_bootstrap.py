"""GPU library bootstrap for TensorFlow on pip-installed CUDA (esp. WSL).

TensorFlow's ``[and-cuda]`` install ships CUDA/cuDNN as ``nvidia-*`` wheels under
``site-packages/nvidia/*/lib``, but the dynamic linker only reads
``LD_LIBRARY_PATH`` at process startup — setting it from within Python is too
late. So if the GPU wheels are present and not yet on the path, we set it and
**re-exec the process once** (guarded by an env flag to avoid loops).

This is a no-op for CPU/ONNX installs (no ``nvidia`` package), so those users pay
nothing.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys

_FLAG = "PERCHLAB_CUDA_BOOTSTRAPPED"
_WSL_DRIVER_DIR = "/usr/lib/wsl/lib"


def ensure_cuda_library_path() -> None:
    """Put bundled CUDA libs on ``LD_LIBRARY_PATH``, re-execing once if needed."""
    if os.environ.get(_FLAG):
        return

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return  # No GPU wheels installed (CPU/ONNX install): nothing to do.

    nvidia_root = next(iter(spec.submodule_search_locations))
    lib_dirs = sorted(glob.glob(os.path.join(nvidia_root, "*", "lib")))
    if not lib_dirs:
        return

    candidates = [*lib_dirs]
    if os.path.isdir(_WSL_DRIVER_DIR):
        candidates.append(_WSL_DRIVER_DIR)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    existing = current.split(":") if current else []
    missing = [d for d in candidates if d not in existing]
    if not missing:
        return  # Already configured.

    os.environ["LD_LIBRARY_PATH"] = ":".join([*missing, *existing]).strip(":")
    os.environ[_FLAG] = "1"
    # Re-exec with the corrected environment so the linker sees the new path.
    os.execv(sys.executable, [sys.executable, *sys.argv])
