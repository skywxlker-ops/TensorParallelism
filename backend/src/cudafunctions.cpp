#include "cudafunctions.hpp"

thread_local static DeviceIndex targetDeviceIndex = -1;

// Internal helper
static int device_count_impl(bool fail_if_no_driver) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice) return 0;
    else if (err == cudaErrorInsufficientDriver) {
        if (fail_if_no_driver) {
            throw std::runtime_error("CUDA driver is missing or too old.");
        }
        return 0;
    } else if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return count;
}

// Public API
int device_count() noexcept {
    try {
        return device_count_impl(false);
    } catch (const std::exception &e) {
        std::cerr << "CUDA warning: " << e.what() << "\n";
        return 0;
    }
}

int device_count_ensure_non_zero() {
    int count = device_count_impl(true);
    if (count == 0) throw std::runtime_error("No CUDA devices available.");
    return count;
}

DeviceIndex current_device() {
    int dev = -1;
    CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

void set_device(DeviceIndex device, bool force) {
    if (device < 0) throw std::runtime_error("Device index must be non-negative.");
    if (force) {
        CUDA_CHECK(cudaSetDevice(device));
        targetDeviceIndex = -1;
        return;
    }

    int cur_device = current_device();
    if (cur_device != device) {
        CUDA_CHECK(cudaSetDevice(device));
        targetDeviceIndex = -1;
    }
}

void MaybeSetDevice(DeviceIndex device) {
    int count = device_count();
    if (device < 0 || device >= count) {
        throw std::runtime_error("Invalid device index in MaybeSetDevice.");
    }
    targetDeviceIndex = device;
}

DeviceIndex ExchangeDevice(DeviceIndex to_device) {
    DeviceIndex old_device = current_device();
    if (to_device != old_device) {
        CUDA_CHECK(cudaSetDevice(to_device));
    }
    targetDeviceIndex = -1;
    return old_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
    DeviceIndex old_device = current_device();
    if (to_device != old_device) {
        if (targetDeviceIndex == to_device) {
            // Do nothing
        } else {
            CUDA_CHECK(cudaSetDevice(to_device));
        }
    }
    return old_device;
}

void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}
