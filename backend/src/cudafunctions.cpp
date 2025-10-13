#include "cudafunctions.hpp"

// Global thread-local target device for MaybeSetDevice
thread_local static DeviceIndex targetDeviceIndex = -1;

// ================= Device Count =================
int device_count_impl(bool fail_if_no_driver) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice) return 0;
    if (err == cudaErrorInsufficientDriver) {
        if (fail_if_no_driver) throw std::runtime_error("CUDA driver missing or too old.");
        return 0;
    }
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return count;
}

DeviceIndex device_count() noexcept {
    try { return device_count_impl(false); } 
    catch (...) { return 0; }
}

DeviceIndex device_count_ensure_non_zero() {
    int count = device_count_impl(true);
    if (count == 0) throw std::runtime_error("No CUDA devices available.");
    return count;
}

// ================= Current Device =================
DeviceIndex current_device() {
    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return dev;
}

// ================= Set Device =================
void set_device(DeviceIndex device, bool force) {
    if (device < 0) throw std::runtime_error("Device index must be non-negative.");
    if (force || current_device() != device) cudaSetDevice(device);
    targetDeviceIndex = -1;
}

// ================= Maybe Set Device =================
void MaybeSetDevice(DeviceIndex device) {
    int count = device_count();
    if (device < 0 || device >= count)
        throw std::runtime_error("Invalid device index in MaybeSetDevice.");
    targetDeviceIndex = device;
}

// ================= Exchange Device =================
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
    DeviceIndex old_device = current_device();
    if (to_device != old_device) cudaSetDevice(to_device);
    targetDeviceIndex = -1;
    return old_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
    DeviceIndex old_device = current_device();
    if (to_device != old_device && targetDeviceIndex != to_device) cudaSetDevice(to_device);
    return old_device;
}

// ================= Device Synchronization =================
void device_synchronize() { cudaDeviceSynchronize(); }