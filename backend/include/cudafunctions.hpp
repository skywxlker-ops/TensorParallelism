#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using DeviceIndex = int;

// ================= Device Count =================
int device_count_impl(bool fail_if_no_driver = false);
DeviceIndex device_count() noexcept;
DeviceIndex device_count_ensure_non_zero();

// ================= Current Device =================
DeviceIndex current_device();

// ================= Set Device =================
void set_device(DeviceIndex device, bool force = false);

// ================= Maybe Set Device =================
void MaybeSetDevice(DeviceIndex device);

// ================= Exchange Device =================
DeviceIndex ExchangeDevice(DeviceIndex to_device);
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device);

// ================= Device Synchronization =================
void device_synchronize();
