using System;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace GpuAccess
{
    /// <summary>
    /// numpy-style elementwise and matrix operations, offloaded to the GPU
    /// through ILGPU (CUDA on NVIDIA, OpenCL on AMD/Intel, CPU fallback
    /// otherwise). No PyTorch, no CuPy - plain float arrays in, plain float
    /// arrays out, so this pairs directly with numpy on the Python side.
    /// </summary>
    public static class GpuArrayOps
    {
        private static readonly object _syncLock = new object();
        private static Context? _context;
        private static Accelerator? _accelerator;

        private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _addKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _subKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _mulKernel;
        private static Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _scaleKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _reluKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _sigmoidKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _tanhKernel;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _sumKernel;
        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>? _matmulKernel;

        private static Accelerator GetAccelerator()
        {
            if (_accelerator != null)
                return _accelerator;

            lock (_syncLock)
            {
                if (_accelerator != null)
                    return _accelerator;

                _context = Context.Create(builder => builder.Default().EnableAlgorithms());
                var device = _context.GetPreferredDevice(preferCPU: false);
                _accelerator = device.CreateAccelerator(_context);
                return _accelerator;
            }
        }

        private static void CheckSameLength(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Input arrays must be the same length.");
        }

        // ---------------- elementwise binary ops ----------------

        private static void AddKernel(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> r)
            => r[i] = a[i] + b[i];

        public static float[] Add(float[] a, float[] b)
        {
            CheckSameLength(a, b);
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufB = accelerator.Allocate1D(b);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _addKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddKernel);
            
            _addKernel(a.Length, bufA.View, bufB.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        private static void SubKernel(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> r)
            => r[i] = a[i] - b[i];

        public static float[] Subtract(float[] a, float[] b)
        {
            CheckSameLength(a, b);
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufB = accelerator.Allocate1D(b);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _subKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SubKernel);
            
            _subKernel(a.Length, bufA.View, bufB.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        private static void MulKernel(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> r)
            => r[i] = a[i] * b[i];

        public static float[] Multiply(float[] a, float[] b)
        {
            CheckSameLength(a, b);
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufB = accelerator.Allocate1D(b);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _mulKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MulKernel);
            
            _mulKernel(a.Length, bufA.View, bufB.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        private static void ScaleKernel(Index1D i, ArrayView<float> a, float s, ArrayView<float> r)
            => r[i] = a[i] * s;

        public static float[] Scale(float[] a, float scalar)
        {
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _scaleKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, float, ArrayView<float>>(ScaleKernel);
            
            _scaleKernel(a.Length, bufA.View, scalar, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        // ---------------- activations ----------------

        private static void ReluKernel(Index1D i, ArrayView<float> a, ArrayView<float> r)
            => r[i] = XMath.Max(0.0f, a[i]);

        public static float[] Relu(float[] a)
        {
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _reluKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(ReluKernel);
            
            _reluKernel(a.Length, bufA.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        private static void SigmoidKernel(Index1D i, ArrayView<float> a, ArrayView<float> r)
            => r[i] = 1.0f / (1.0f + XMath.Exp(-a[i]));

        public static float[] Sigmoid(float[] a)
        {
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _sigmoidKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(SigmoidKernel);
            
            _sigmoidKernel(a.Length, bufA.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        private static void TanhKernel(Index1D i, ArrayView<float> a, ArrayView<float> r)
            => r[i] = XMath.Tanh(a[i]);

        public static float[] Tanh(float[] a)
        {
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufR = accelerator.Allocate1D<float>(a.Length);
            
            _tanhKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(TanhKernel);
            
            _tanhKernel(a.Length, bufA.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D();
        }

        // ---------------- reduction ----------------

        private static void SumKernel(Index1D i, ArrayView<float> a, ArrayView<float> result)
            => Atomic.Add(ref result[0], a[i]);

        public static float Sum(float[] a)
        {
            var accelerator = GetAccelerator();
            using var bufA = accelerator.Allocate1D(a);
            using var bufR = accelerator.Allocate1D<float>(1);
            bufR.CopyFromCPU(new float[] { 0f });
            
            _sumKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(SumKernel);
            
            _sumKernel(a.Length, bufA.View, bufR.View);
            accelerator.Synchronize();
            return bufR.GetAsArray1D()[0];
        }

        // ---------------- matrix multiply ----------------

        private static void MatMulKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> a,
            ArrayView2D<float, Stride2D.DenseX> b,
            ArrayView2D<float, Stride2D.DenseX> r)
        {
            float sum = 0f;
            var inner = a.IntExtent.Y;
            for (int k = 0; k < inner; k++)
                sum += a[index.X, k] * b[k, index.Y];
            r[index.X, index.Y] = sum;
        }

        /// <summary>
        /// Multiplies a (aRows x aCols) by b (aCols x bCols). Both inputs
        /// and the output are row-major flattened float arrays, so the
        /// Python side can pass numpy arrays straight through after
        /// .ravel().
        /// </summary>
        public static float[] MatMul(float[] aFlat, int aRows, int aCols, float[] bFlat, int bRows, int bCols)
        {
            if (aCols != bRows)
                throw new ArgumentException($"Shape mismatch: ({aRows}x{aCols}) x ({bRows}x{bCols})");
            if (aFlat.Length != aRows * aCols || bFlat.Length != bRows * bCols)
                throw new ArgumentException("Flattened array length does not match given shape.");

            var accelerator = GetAccelerator();

            var aMat = new float[aRows, aCols];
            Buffer.BlockCopy(aFlat, 0, aMat, 0, aFlat.Length * sizeof(float));
            var bMat = new float[bRows, bCols];
            Buffer.BlockCopy(bFlat, 0, bMat, 0, bFlat.Length * sizeof(float));

            using var bufA = accelerator.Allocate2DDenseX(aMat);
            using var bufB = accelerator.Allocate2DDenseX(bMat);
            using var bufR = accelerator.Allocate2DDenseX<float>(new Index2D(aRows, bCols));

            _matmulKernel ??= accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(MatMulKernel);

            _matmulKernel(new Index2D(aRows, bCols), bufA.View, bufB.View, bufR.View);
            accelerator.Synchronize();

            var resultMat = bufR.GetAsArray2D();
            var resultFlat = new float[aRows * bCols];
            Buffer.BlockCopy(resultMat, 0, resultFlat, 0, resultFlat.Length * sizeof(float));
            return resultFlat;
        }
    }
}
