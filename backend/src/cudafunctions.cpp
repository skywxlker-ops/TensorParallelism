#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <optional>

using DeviceIndex = int;

// Helper macros for error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            throw std::runtime_error(cudaGetErrorString(err));   \
        }                                                        \
    } while (0)

// Global thread-local target device for MaybeSetDevice
thread_local static DeviceIndex targetDeviceIndex = -1;

// ================= Device Count =================
int device_count_impl(bool fail_if_no_driver) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice) {
        return 0;
    } else if (err == cudaErrorInsufficientDriver) {
        if (fail_if_no_driver) {
            throw std::runtime_error("CUDA driver is missing or too old.");
        }
        return 0;
    } else if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return count;
}

DeviceIndex device_count() noexcept {
    try {
        return device_count_impl(false);
    } catch (const std::exception &e) {
        std::cerr << "CUDA warning: " << e.what() << "\n";
        return 0;
    }
}

DeviceIndex device_count_ensure_non_zero() {
    int count = device_count_impl(true);
    if (count == 0) throw std::runtime_error("No CUDA devices available.");
    return count;
}

// ================= Current Device =================
DeviceIndex current_device() {
    int dev = -1;
    CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

// ================= Set Device =================
void set_device(DeviceIndex device, bool force = false) {
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

// ================= Maybe Set Device =================
void MaybeSetDevice(DeviceIndex device) {
    int count = device_count();
    if (device < 0 || device >= count) {
        throw std::runtime_error("Invalid device index in MaybeSetDevice.");
    }
    targetDeviceIndex = device; // Do not create context yet
}

// ================= Exchange Device =================
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
            // Do nothing, context not created
        } else {
            CUDA_CHECK(cudaSetDevice(to_device));
        }
    }
    return old_device;
}

// ================= Device Synchronization =================
void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ================= Demo Main =================
int main() {
    try {
        std::cout << "CUDA devices available: " << device_count() << "\n";
        DeviceIndex dev = current_device();
        std::cout << "Current device: " << dev << "\n";

        std::cout << "Switching to device 0...\n";
        set_device(1);

        std::cout << "Maybe setting device 1 (no context created yet)...\n";
        MaybeSetDevice(1);

        std::cout << "Exchanging device with 0...\n";
        DeviceIndex old = ExchangeDevice(0);
        std::cout << "Old device was: " << old << "\n";

        std::cout << "Synchronizing device...\n";
        device_synchronize();
        std::cout << "Done!\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}