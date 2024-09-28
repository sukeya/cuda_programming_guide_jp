#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

int main() {
    // Instruct CUDA to yield its thread when waiting for results from the device.
    unsigned int device_flags = cudaDeviceScheduleYield;
    // Tell the CUDA runtime that DeviceFlags is being set in cudaInitDevice call 
    unsigned int flags = cudaInitDeviceFlagsAreValid;
    // Initialize device to be used for GPU executions.
    auto e = cudaInitDevice(0, device_flags, flags);
    assert(e == cudaSuccess);

    // Set device to be used for GPU executions.
    e = cudaSetDevice(0);
    assert(e == cudaSuccess);
    
    int device;
    e = cudaGetDevice(&device);
    assert(e == cudaSuccess);
    assert(device == 0);

    unsigned int current_flags;
    e = cudaGetDeviceFlags(&current_flags);
    assert(e == cudaSuccess);
    // Flags returned by this function may specifically include cudaDeviceMapHost
    // even though it is not accepted by cudaSetDeviceFlags because it is implicit
    // in runtime API flags.
    assert((current_flags & device_flags) == device_flags);
}
