using System;
using ILGPU;
using ILGPU.Runtime;

namespace GpuAccess
{
    /// <summary>
    /// Minimal example of running real work on the GPU (or best available
    /// accelerator). Exposes plain array-in / array-out methods so they are
    /// easy to call from Python through pythonnet.
    /// </summary>
    public static class GpuCompute
    {
        private static void VectorAddKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> result)
        {
            result[index] = a[index] + b[index];
        }

        /// <summary>
        /// Adds two float arrays element-wise on the GPU. Automatically
        /// falls back to a CPU accelerator if no GPU is available, so it
        /// always returns a result.
        /// </summary>
        public static float[] VectorAdd(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Input arrays must be the same length.");

            using var context = Context.Create(builder => builder.Default().EnableAlgorithms());
            var device = context.GetPreferredDevice(preferCPU: false);
            using var accelerator = device.CreateAccelerator(context);

            using var bufferA = accelerator.Allocate1D(a);
            using var bufferB = accelerator.Allocate1D(b);
            using var bufferResult = accelerator.Allocate1D<float>(a.Length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

            kernel(a.Length, bufferA.View, bufferB.View, bufferResult.View);
            accelerator.Synchronize();

            return bufferResult.GetAsArray1D();
        }

        /// <summary>
        /// Name of the accelerator that compute calls will actually run on.
        /// Useful to confirm whether a real GPU or the CPU fallback is active.
        /// </summary>
        public static string GetActiveDeviceName()
        {
            using var context = Context.Create(builder => builder.Default());
            var device = context.GetPreferredDevice(preferCPU: false);
            return $"{device.Name} ({device.AcceleratorType})";
        }
    }
}
