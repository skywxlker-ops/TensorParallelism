# TensorParallel Backend

A minimal **Tensor Parallel (TP) backend** using CUDA + NCCL, providing multi-GPU mesh abstraction and collective operations.

---

## 📦 Structure

include/ # Mesh & Task headers
src/ # Mesh & Task implementations
tests/ # Test executables


---

## 🔹 Core Components

### Mesh
- Initializes NCCL communicators & CUDA streams across GPUs.
- Supports multi-GPU collectives (AllReduce).

```cpp
Mesh mesh(num_gpus);
mesh.allReduce(device_ptr, num_elements);

Task

    Grid-stride kernel for tensor initialization.

    Multi-threaded AllReduce to avoid deadlocks.

std::vector<float*> d_data(num_gpus);
Task::initTensors(d_data, mesh, num_elements);
Task::runAllReduce(mesh, d_data, num_elements);

 Improvements
Issue	Fix
NCCL hangs	Multi-threaded AllReduce
Illegal memory access	Synchronize kernels & streams
Tiny tensors	Increased size (1024 elements)
Memory leaks	Proper cleanup of streams & NCCL comms
 Usage

nvcc -std=c++17 -Iinclude -lnccl -lpthread -o tests/test_task src/mesh.cu src/task.cu tests/test_task.cpp
./tests/test_task

Expected output:

[Mesh] Initializing mesh with 2 GPUs...
[Task] Performing AllReduce across 2 GPUs...
[GPU 0] Output: 3 3 3 3 ...
[GPU 1] Output: 3 3 3 3 ...
[Mesh] Destroyed NCCL communicator.

 Next Steps

    Build a Tensor abstraction (shape, placement, mesh info).

    Integrate sharding & placement logic.

    Implement advanced collectives (ReduceScatter, AllGather).

    Develop TP-aware model layers (Linear, MatMul, Softmax).
