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

## Build & Run

```bash
cd backend
./build.sh                 # Compile all sources
./build/test_dtensor       # Test DTensor
./build/test_mesh          # Test mesh layouts
./build/test_logical_gpu   # Test logical GPU functionality
./build/test_logical_nccl  # Test hybrid NCCL simulation
./build/test_task          # Test task execution


