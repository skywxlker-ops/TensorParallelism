# TensorParallelism Backend

This repository contains a minimal framework for **multi-GPU tensor parallelism** using CUDA and NCCL.  
It includes:

- Device utilities (`cudafunctions`) for safe GPU handling.
- `Mesh` abstraction with topology awareness and device subgroups.
- `Task` utilities for initializing tensors and performing AllReduce.
- Tests demonstrating Mesh and Task functionality.

---

## Directory Structure

tp/backend/
├── include/
│ ├── cudafunctions.hpp
│ ├── mesh.hpp
│ └── task.hpp
├── src/
│ ├── cudafunctions.cpp
│ ├── mesh.cu
│ └── task.cu
└── tests/
├── test_mesh.cpp
└── test_task.cpp


---

## **1. Device Utilities (`cudafunctions`)**

`cudafunctions` provides:

- Detecting the number of CUDA devices:
```cpp
int device_count();
int device_count_ensure_non_zero();

    Current device and switching:

DeviceIndex current_device();
void set_device(DeviceIndex device, bool force=false);
DeviceIndex ExchangeDevice(DeviceIndex to_device);
void MaybeSetDevice(DeviceIndex device);

    Synchronization:

void device_synchronize();

Example usage:

std::cout << "CUDA devices available: " << device_count() << "\n";
DeviceIndex dev = current_device();
set_device(1);
device_synchronize();

2. Mesh Abstraction

Mesh manages multiple GPUs with:

    Topology awareness:

std::vector<int64_t> mesh_shape_;                // mesh shape
std::map<int, std::vector<int>> mesh_coords_;   // GPU logical coords

mesh.setMeshShape({2});  // 1D mesh with 2 GPUs

    Device subgroups:

mesh.createSubGroup("tensor", {0,1});
ncclComm_t comm = mesh.getSubComm("tensor");

    NCCL communicators and CUDA streams are automatically initialized.

Example:

Mesh mesh(num_gpus);
mesh.setMeshShape({2});
mesh.createSubGroup("tensor", {0,1});

Output:

[Mesh] Mesh shape set to [2]
[Mesh] GPU 0 logical coords: [0]
[Mesh] GPU 1 logical coords: [1]
[Mesh] Creating subgroup 'tensor' with devices: 0 1

3. Task Utilities

Task provides:

    Tensor initialization on multiple GPUs:

Task::initTensors(d_data, mesh, num_elements);

    AllReduce across all GPUs (or subgroups):

Task::runAllReduce(mesh, d_data, num_elements);

Example:

std::vector<float*> d_data(num_gpus);
Task::initTensors(d_data, mesh, 1024);
Task::runAllReduce(mesh, d_data, 1024);

Sample output:

[Task] Performing AllReduce across 2 GPUs...
[GPU 0] Output: 3 3 3 3 3 3 3 3 3 3 ...
[GPU 1] Output: 3 3 3 3 3 3 3 3 3 3 ...

4. Compilation & Running Tests

Compile test_mesh:

nvcc -Iinclude -o tests/test_mesh tests/test_mesh.cpp src/mesh.cu src/cudafunctions.cpp -lnccl

Compile test_task:

nvcc -Iinclude -o tests/test_task tests/test_task.cpp src/mesh.cu src/task.cu src/cudafunctions.cpp -lnccl

Run:

./tests/test_mesh
./tests/test_task

Expected output for 2 GPUs:

[Mesh] Initializing mesh with 2 GPUs...
[Mesh] NCCL communicators initialized successfully.
[Mesh] Mesh shape set to [2]
[Mesh] GPU 0 logical coords: [0]
[Mesh] GPU 1 logical coords: [1]
[Mesh] Creating subgroup 'tensor' with devices: 0 1
[Task] Performing AllReduce across 2 GPUs...
[GPU 0] Output: 3 3 3 3 3 3 3 3 3 3 ...
[GPU 1] Output: 3 3 3 3 3 3 3 3 3 3 ...
[Mesh] Destroyed NCCL communicator.

---
## Flow - 

[CUDA Devices]
       │
       ▼
[cudafunctions.cpp]
   ├─ device_count() / device_count_ensure_non_zero()
   ├─ current_device()
   ├─ set_device(), ExchangeDevice(), MaybeSetDevice()
   └─ device_synchronize()
       │
       ▼
[Mesh (mesh.hpp / mesh.cu)]
   ├─ Initialize NCCL communicators & CUDA streams for each GPU
   ├─ Set mesh_shape_ → compute mesh_coords_ (logical coords)
   │      e.g., GPU 0 → [0], GPU 1 → [1]
   ├─ createSubGroup(name, devices) → define logical subgroups
   │      e.g., "tensor" subgroup: {0,1}
   └─ getSubComm(name) → returns communicator for subgroup
       │
       ▼
[Task (task.hpp / task.cu)]
   ├─ initTensors(d_data, mesh, num_elements)
   │      ├─ Allocate memory on each GPU using set_device()
   │      └─ Initialize tensor values via initTensorKernel
   └─ runAllReduce(mesh, d_data, num_elements)
          ├─ Launch threads per GPU
          ├─ Use Mesh subcommunicator or global communicator
          ├─ Perform NCCL AllReduce
          └─ Synchronize and copy results to host
       │
       ▼
[Output / Verification]
   ├─ Print logical coordinates
   ├─ Print subgroup info
   └─ Print AllReduce results (sum across GPUs)

---