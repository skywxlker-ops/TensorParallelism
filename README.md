# Tensor Parallelism Backend – Summary

A lightweight multi-GPU tensor parallelism framework using **CUDA + NCCL**.

---

## **Key Components**

### **1. Device Utilities (`cudafunctions`)**
- Detect GPUs: `device_count()`, `device_count_ensure_non_zero()`
- Manage devices: `current_device()`, `set_device()`, `ExchangeDevice()`, `MaybeSetDevice()`
- Synchronize: `device_synchronize()`
- Ensures safe and consistent GPU handling across threads and tasks.

### **2. Mesh**
- Represents a set of GPUs.
- Initializes **NCCL communicators** and **CUDA streams** per GPU.
- Provides helper functions for collective operations (e.g., `allReduce`).

### **3. Task**
- Initializes tensors on each GPU.
- Performs **AllReduce** across the mesh concurrently using threads.
- Prints results for verification.

---

## **Directory Structure**

backend/
include/ # mesh.hpp, task.hpp, cudafunctions.hpp
src/ # mesh.cu, task.cu, cudafunctions.cpp
tests/ # test_mesh.cpp, test_task.cpp


---

## **How to Build & Run**

### Compile
```bash
nvcc -Iinclude -o test_mesh tests/test_mesh.cpp src/cudafunctions.cpp src/mesh.cu -lnccl
nvcc -Iinclude -o test_task tests/test_task.cpp src/cudafunctions.cpp src/mesh.cu src/task.cu -lnccl

Run

./test_mesh
./test_task

Sample Output (2 GPUs)

test_mesh

[Mesh] Initializing mesh with 2 GPUs...
[Mesh] NCCL communicators initialized successfully.
[Mesh] Destroyed NCCL communicator.

test_task

[Task] Performing AllReduce across 2 GPUs...
[GPU 0] Output: 3 3 3 3 3 3 3 3 3 3 ...
[GPU 1] Output: 3 3 3 3 3 3 3 3 3 3 ...

    Values show summed tensors across GPUs after AllReduce.