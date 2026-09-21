using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;

namespace GpuAccess
{
    /// <summary>
    /// Reports information about the GPUs (and other accelerators) visible
    /// on this machine. Works across CUDA (NVIDIA), OpenCL (AMD / Intel /
    /// NVIDIA) and falls back to a CPU accelerator if no real GPU is found,
    /// so calling code never has to special-case the hardware.
    /// </summary>
    public static class GpuInfo
    {
        public class DeviceInfo
        {
            public int Index;
            public string Name = "";
            public string AcceleratorType = "";
            public long MemorySizeBytes;
            public int MaxNumThreadsPerGroup;
            public bool IsDefault;
        }

        /// <summary>
        /// Lists every accelerator ILGPU can see on this machine.
        /// </summary>
        public static List<DeviceInfo> ListDevices()
        {
            var results = new List<DeviceInfo>();

            using var context = Context.Create(builder => builder.AllAccelerators());
            var devices = context.Devices.ToList();

            for (int i = 0; i < devices.Count; i++)
            {
                var device = devices[i];
                results.Add(new DeviceInfo
                {
                    Index = i,
                    Name = device.Name,
                    AcceleratorType = device.AcceleratorType.ToString(),
                    MemorySizeBytes = device.MemorySize,
                    MaxNumThreadsPerGroup = device.MaxNumThreadsPerGroup,
                    IsDefault = i == 0
                });
            }

            return results;
        }

        /// <summary>
        /// Short human readable summary, one line per device. Handy for a
        /// quick print() from Python without dealing with objects.
        /// </summary>
        public static string[] ListDevicesAsStrings()
        {
            return ListDevices()
                .Select(d => $"[{d.Index}] {d.Name} ({d.AcceleratorType}) - " +
                             $"{d.MemorySizeBytes / (1024 * 1024)} MB")
                .ToArray();
        }
    }
}
