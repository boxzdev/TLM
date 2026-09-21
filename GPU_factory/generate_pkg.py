"""
generate_pkg.py
===============
Ensures that cugpu_pkg and all required Python source files
(pyproject.toml, README.md, cugpu/__init__.py) exist.
If cugpu_pkg was deleted, running this file (or build_package.bat)
will automatically recreate everything from scratch.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT_DIR, "cugpu_pkg")
SRC_DIR = os.path.join(PKG_DIR, "cugpu")

PYPROJECT_CONTENT = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cugpu"
version = "0.1.0"
description = "Python package that accesses the GPU through a C# / ILGPU backend"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "pythonnet>=3.0.3",
    "numpy"
]

[tool.setuptools]
packages = ["cugpu"]

[tool.setuptools.package-data]
cugpu = ["_native/*.dll"]
"""

README_CONTENT = """# cugpu

A Python package that reaches the GPU through a compiled C# (.NET) library
(`GpuAccess.dll`), loaded at runtime via [pythonnet](https://pythonnet.github.io/).

The C# side uses [ILGPU](https://github.com/m4rs-mt/ILGPU), so it works over
OpenCL (AMD / Intel / NVIDIA), CUDA (NVIDIA), or a CPU accelerator fallback
if no GPU driver is found — no vendor-specific code needed.

## Usage

```python
import cugpu

cugpu.print_gpus()                      # list every accelerator found
print(cugpu.active_device())            # which one compute calls use
print(cugpu.vector_add([1, 2, 3], [4, 5, 6]))   # [5.0, 7.0, 9.0]
```

## Note

This folder is only the Python side. It expects `GpuAccess.dll` (and its
dependencies) to already be sitting in `cugpu/_native/`. That's done
automatically by `build_package.bat` in the parent `GPU_factory` folder,
which compiles the C# project first and copies the output here before
building this wheel.
"""

INIT_PY_CONTENT = '''"""
cugpu
=====
A thin Python wrapper around a C# (.NET) library that talks to the GPU
through ILGPU. Uses pythonnet to host the .NET runtime inside the Python
process and call straight into the compiled C# assembly (GpuAccess.dll).

Typical usage:

    import cugpu as cg

    cg.print_gpus()               # list every accelerator found
    print(cg.active_device())     # which one compute calls will use
    result = cg.vector_add([1, 2, 3], [4, 5, 6])
"""

import os
import sys

import numpy as np

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_PACKAGE_DIR, "_native")
_DLL_NAME = "GpuAccess.dll"

_clr_loaded = False
GpuInfo = None
GpuCompute = None
GpuArrayOps = None


def _ensure_loaded():
    """Lazily boots the .NET runtime and loads GpuAccess.dll the first time
    it's actually needed, instead of paying that cost at import time."""
    global _clr_loaded, GpuInfo, GpuCompute, GpuArrayOps

    if _clr_loaded:
        return

    dll_path = os.path.join(_NATIVE_DIR, _DLL_NAME)
    if not os.path.isfile(dll_path):
        raise FileNotFoundError(
            f"Could not find {_DLL_NAME} in {_NATIVE_DIR}.\\n"
            "Run build_package.bat first so the C# project gets compiled "
            "and its DLLs get copied into this package."
        )

    try:
        from clr_loader import get_coreclr
        from pythonnet import set_runtime
        try:
            set_runtime(get_coreclr())
        except Exception:
            pass
        import clr  # provided by the 'pythonnet' package
    except ImportError as exc:
        raise ImportError(
            "cugpu needs the 'pythonnet' package to talk to the C# "
            "GpuAccess.dll. Install it with: pip install pythonnet"
        ) from exc

    if _NATIVE_DIR not in sys.path:
        sys.path.append(_NATIVE_DIR)

    clr.AddReference("GpuAccess")
    from GpuAccess import GpuInfo as _GpuInfo
    from GpuAccess import GpuCompute as _GpuCompute
    from GpuAccess import GpuArrayOps as _GpuArrayOps

    GpuInfo = _GpuInfo
    GpuCompute = _GpuCompute
    GpuArrayOps = _GpuArrayOps
    _clr_loaded = True


def list_gpus():
    """Returns a list of dicts describing every accelerator ILGPU can see
    (real GPUs first, CPU fallback last if no GPU driver is present)."""
    _ensure_loaded()
    devices = GpuInfo.ListDevices()
    return [
        {
            "index": d.Index,
            "name": d.Name,
            "type": str(d.AcceleratorType),
            "memory_mb": d.MemorySizeBytes // (1024 * 1024),
            "max_threads_per_group": d.MaxNumThreadsPerGroup,
            "is_default": bool(d.IsDefault),
        }
        for d in devices
    ]


def print_gpus():
    """Quick human readable printout of available accelerators."""
    _ensure_loaded()
    for line in GpuInfo.ListDevicesAsStrings():
        print(line)


def active_device():
    """Name of the accelerator that compute calls will actually run on."""
    _ensure_loaded()
    return GpuCompute.GetActiveDeviceName()


def vector_add(a, b):
    """Adds two equal-length lists/arrays of numbers on the GPU and returns
    a plain Python list of floats."""
    _ensure_loaded()
    result = GpuCompute.VectorAdd(list(map(float, a)), list(map(float, b)))
    return list(result)


# ---------------------------------------------------------------------
# numpy-friendly array ops (add / relu / matmul / etc.) - the actual
# building blocks for writing a neural net from scratch without
# PyTorch or CuPy. Every function below takes numpy arrays (or plain
# lists) and returns numpy arrays.
# ---------------------------------------------------------------------

def _flatten_f32(x):
    arr = np.asarray(x, dtype=np.float32)
    return arr.shape, arr.ravel().tolist()


def add(a, b):
    """Elementwise a + b on the GPU. Returns a numpy array shaped like a."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    _, flat_b = _flatten_f32(b)
    result = GpuArrayOps.Add(flat_a, flat_b)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def subtract(a, b):
    """Elementwise a - b on the GPU. Returns a numpy array shaped like a."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    _, flat_b = _flatten_f32(b)
    result = GpuArrayOps.Subtract(flat_a, flat_b)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def multiply(a, b):
    """Elementwise a * b on the GPU. Returns a numpy array shaped like a."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    _, flat_b = _flatten_f32(b)
    result = GpuArrayOps.Multiply(flat_a, flat_b)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def scale(a, scalar):
    """Multiplies every element of a by a Python float, on the GPU."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    result = GpuArrayOps.Scale(flat_a, float(scalar))
    return np.array(list(result), dtype=np.float32).reshape(shape)


def relu(a):
    """ReLU activation, elementwise, on the GPU."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    result = GpuArrayOps.Relu(flat_a)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def sigmoid(a):
    """Sigmoid activation, elementwise, on the GPU."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    result = GpuArrayOps.Sigmoid(flat_a)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def tanh(a):
    """Tanh activation, elementwise, on the GPU."""
    _ensure_loaded()
    shape, flat_a = _flatten_f32(a)
    result = GpuArrayOps.Tanh(flat_a)
    return np.array(list(result), dtype=np.float32).reshape(shape)


def gpu_sum(a):
    """Sums every element of a on the GPU, returns a Python float."""
    _ensure_loaded()
    _, flat_a = _flatten_f32(a)
    return float(GpuArrayOps.Sum(flat_a))


def matmul(a, b):
    """Matrix multiply two 2D numpy arrays on the GPU: (m x k) @ (k x n) -> (m x n)."""
    _ensure_loaded()
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError("matmul expects 2D arrays")

    a_rows, a_cols = arr_a.shape
    b_rows, b_cols = arr_b.shape
    result = GpuArrayOps.MatMul(
        arr_a.ravel().tolist(), a_rows, a_cols,
        arr_b.ravel().tolist(), b_rows, b_cols,
    )
    return np.array(list(result), dtype=np.float32).reshape(a_rows, b_cols)


__all__ = [
    "list_gpus", "print_gpus", "active_device", "vector_add",
    "add", "subtract", "multiply", "scale", "relu", "sigmoid", "tanh",
    "gpu_sum", "matmul",
]
'''


def ensure_package_sources():
    os.makedirs(SRC_DIR, exist_ok=True)
    targets = {
        os.path.join(PKG_DIR, "pyproject.toml"): PYPROJECT_CONTENT,
        os.path.join(PKG_DIR, "README.md"): README_CONTENT,
        os.path.join(SRC_DIR, "__init__.py"): INIT_PY_CONTENT,
    }

    created_any = False
    for path, content in targets.items():
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [+] Created missing source file: {os.path.relpath(path, ROOT_DIR)}")
            created_any = True
        else:
            print(f"  [=] Found existing source file: {os.path.relpath(path, ROOT_DIR)}")

    if not created_any:
        print("  All Python package source files are present.")


if __name__ == "__main__":
    ensure_package_sources()
