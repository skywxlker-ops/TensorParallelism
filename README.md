# TensorParallelism Backend

A custom framework for distributed tensor computations and tensor parallelism on multi-GPU systems. Supports dynamic tensor slicing, sharding, replication, partial placements, and hybrid logical GPU simulations.

---

## Features

- **DTensor**
  - Dynamic tensor slicing across GPUs
  - Placement types:
    - `Shard` – tensor split across GPUs
    - `Replicate` – full copy on all GPUs
    - `Partial` – hybrid sharding and replication
  - `printSlices()` to inspect GPU slices

- **DeviceMesh**
  - N-dimensional logical mesh abstraction
  - Maps logical GPUs to physical GPUs
  - Supports 1D, 2D, and 3D layouts

- **Logical GPUs**
  - Simulate multiple GPUs per physical device
  - Independent CUDA streams and memory regions
  - Enables testing larger mesh topologies on limited hardware

- **Hybrid NCCL Simulation**
  - Local collectives on logical GPUs sharing a physical GPU
  - Real NCCL collectives across physical GPUs
  - Supports testing 2D/3D mesh collectives without extra hardware

---

## Directory  

```bash
.
├── backend
│   ├── build
│   │   ├── test_dtensor
│   │   ├── test_logical_gpu
│   │   ├── test_logical_nccl
│   │   ├── test_mesh
│   │   └── test_task
│   ├── include
│   │   ├── cudafunctions.hpp
│   │   ├── dtensor.hpp
│   │   ├── logical_gpu.hpp
│   │   ├── mesh.hpp
│   │   └── task.hpp
│   ├── src
│   │   ├── cudafunctions.cpp
│   │   ├── dtensor.cu
│   │   ├── dtensor_host.cpp
│   │   ├── logical_gpu.cu
│   │   ├── logical_gpu_kernel.cu
│   │   ├── mesh.cu
│   │   └── task.cu
│   ├── tests
│   │   ├── test_dtensor.cpp
│   │   ├── test_logical_gpu.cu
│   │   ├── test_logical_nccl.cpp
│   │   ├── test_mesh.cpp
│   │   └── test_task.cpp
│   └── build.sh
├── README.md
├── show_dir.sh
└── update_github.sh
---
---
## Build & Run

```bash
cd backend
./build.sh                 # Compile all sources
./build/test_dtensor       # Test DTensor
./build/test_mesh          # Test mesh layouts
./build/test_logical_gpu   # Test logical GPU functionality
./build/test_logical_nccl  # Test hybrid NCCL simulation
./build/test_task          # Test task execution


