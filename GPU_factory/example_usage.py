r"""
Run this AFTER build_package.bat has finished and you've
pip-installed the resulting wheel:

    pip install cugpu_pkg\dist\cugpu-0.1.0-py3-none-any.whl
    python example_usage.py
"""

import numpy as np
import cugpu

print("Accelerators found:")
cugpu.print_gpus()

print("\nActive device for compute calls:")
print(" ", cugpu.active_device())

# --- numpy-style ops, no PyTorch / CuPy involved ---
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0], dtype=np.float32)

print("\nGPU add:", cugpu.add(a, b))
print("GPU relu(-1, 0, 5):", cugpu.relu(np.array([-1.0, 0.0, 5.0])))
print("GPU sigmoid(0):", cugpu.sigmoid(np.array([0.0])))
print("GPU sum:", cugpu.gpu_sum(a))

# a tiny "layer": out = relu(W @ x)
W = np.random.randn(4, 3).astype(np.float32)
x = np.random.randn(3, 1).astype(np.float32)
out = cugpu.relu(cugpu.matmul(W, x))
print("\nW @ x, then ReLU, shape:", out.shape)
print(out)
