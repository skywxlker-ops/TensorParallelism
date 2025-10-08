#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using DeviceIndex = int;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Core device utilities
int device_count() noexcept;
int device_count_ensure_non_zero();
DeviceIndex current_device();
void set_device(DeviceIndex device, bool force = false);
void MaybeSetDevice(DeviceIndex device);
DeviceIndex ExchangeDevice(DeviceIndex to_device);
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device);
void device_synchronize();
