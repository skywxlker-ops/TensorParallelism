# TensorParallelism Backend

This is a lightweight multi-GPU Tensor Parallelism backend using **CUDA + NCCL**.

## Features
- GPU mesh creation (Mesh class)
- NCCL communicator initialization
- Task abstraction for distributed operations
- Example AllReduce task across multiple GPUs
