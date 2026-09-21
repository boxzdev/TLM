# cugpu

A Python package that talks to AMD, Intel, and NVIDIA GPUs through a C# / ILGPU backend.

```
GPU_factory/
├── package_factory/          <- all C# source (the GPU-access library)
│   ├── GpuAccess.csproj
│   ├── GpuInfo.cs             (lists available GPUs / accelerators)
│   ├── GpuCompute.cs          (example: runs a vector-add kernel on the GPU)
│   └── GpuArrayOps.cs         (numpy-style ops: add/sub/mul/scale, relu,
│                                sigmoid, sum, matmul - for AI without
│                                PyTorch or CuPy)
│
├── cugpu_pkg/                 <- Python package (auto-generated if deleted!)
│   ├── pyproject.toml
│   ├── README.md
│   └── cugpu/
│       ├── __init__.py        (loads GpuAccess.dll via pythonnet)
│       └── _native/           (compiled DLLs get copied here by the .bat)
│
├── generate_pkg.py            <- auto-creates cugpu_pkg and all 3 source files
├── build_package.bat          <- outside package_factory: builds everything
├── example_usage.py
└── README.md
```

## How it works

- **C# side (`package_factory/`)**: uses [ILGPU](https://github.com/m4rs-mt/ILGPU),
  a .NET library that runs compute kernels on OpenCL (AMD / Intel / NVIDIA),
  CUDA (NVIDIA), or a CPU accelerator if no GPU is present. `GpuInfo.cs`
  lists the accelerators it finds; `GpuCompute.cs` runs an example kernel.
- **Python side (`cugpu_pkg/`)**: a normal pip package. Its
  `__init__.py` uses [pythonnet](https://pythonnet.github.io/) to host the
  .NET runtime inside the Python process and call directly into the
  compiled `GpuAccess.dll` — no separate server process, no sockets.
- **`build_package.bat`**: compiles the C# project, copies the resulting
  DLLs into `cugpu_pkg/cugpu/_native/`, then runs
  `python -m build` to produce an installable wheel.

## Requirements

- Windows, with:
  - [.NET SDK 8.0+](https://dotnet.microsoft.com/download)
  - Python 3.8+ on PATH
- A GPU + driver is **not required** — ILGPU falls back to a CPU
  accelerator automatically, so the package always works, it's just faster
  with a real GPU.

## GPU vendor support

ILGPU has two GPU backends:

- **OpenCL** — AMD and Intel GPUs (and NVIDIA too), as long as the
  vendor's normal graphics driver is installed (AMD Adrenalin, Intel
  graphics driver — both ship an OpenCL runtime by default).
- **CUDA** — NVIDIA GPUs.

`GetPreferredDevice(preferCPU: false)` in the C# code automatically picks
whichever real GPU it finds first; if none is found it uses the CPU
accelerator instead, so nothing crashes on a machine without a GPU.

This is the opposite tradeoff from **CuPy**, which only supports NVIDIA
CUDA (plus limited AMD ROCm on Linux) and has **no Intel GPU support at
all**. If you need AMD or Intel GPUs specifically, this ILGPU-based
approach is the one that actually covers them.

## AI building blocks (no PyTorch, no CuPy)

`cugpu` exposes plain numpy-in / numpy-out functions, all running on
the GPU through the C# backend:

```python
import numpy as np
import cugpu

cugpu.add(a, b)          # elementwise add
cugpu.subtract(a, b)
cugpu.multiply(a, b)
cugpu.scale(a, 2.0)
cugpu.relu(a)
cugpu.sigmoid(a)
cugpu.tanh(a)
cugpu.gpu_sum(a)
cugpu.matmul(W, x)        # (m x k) @ (k x n) -> (m x n)
```

That's enough to hand-write a small neural network's forward pass (matmul
+ bias add + activation, layer by layer) using only numpy arrays as the
data structure — no PyTorch, no CuPy.

## Build

Double-click `build_package.bat`, or from a terminal:

```bat
build_package.bat
```

This produces a wheel at `cugpu_pkg\dist\cugpu-0.1.0-py3-none-any.whl`.

## Install & use

```bat
pip install cugpu_pkg\dist\cugpu-0.1.0-py3-none-any.whl
python example_usage.py
```

or

```bat
pip install GPU_factory\cugpu_pkg
```

```python
import cugpu

cugpu.print_gpus()                 # e.g. [1] gfx90c (OpenCL) - 3125 MB
print(cugpu.active_device())
print(cugpu.vector_add([1, 2, 3], [4, 5, 6]))   # [5.0, 7.0, 9.0]
```

## Extending it

Add more GPU logic as new `.cs` files in `package_factory/` (any public
static methods work), then expose thin Python wrappers for them in
`cugpu_pkg/cugpu/__init__.py`. Re-run `build_package.bat` to
rebuild the DLL and the wheel together.
