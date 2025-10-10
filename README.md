TensorParallelism Backend

A minimal CUDA + NCCL backend simulating core components of tensor parallelism: device mesh, distributed tensor (DTensor), placements, and communication.

📂 Structure
backend/
├── include/
│   ├── mesh.hpp
│   ├── cudafunctions.hpp
│   ├── dtensor.hpp
├── src/
│   ├── mesh.cu
│   ├── cudafunctions.cpp
│   ├── dtensor.cu
├── tests/
│   ├── test_mesh.cpp
│   ├── test_dtensor.cpp
├── build.sh
└── README.md


Build Instructions

chmod +x build.sh
./build.sh



Executables are created under build/:

./build/test_mesh
./build/test_dtensor



🚀 Execution Flow

Mesh Initialization

Sets up GPU devices and NCCL communicators.

Builds logical mesh coordinates.

DTensor Creation

A global tensor (e.g., 8×4) is divided across GPUs based on placements.

Example placement:

GPU 0 → rows [0,3], columns [0,3]

GPU 1 → rows [4,7], columns [0,3]

Printing and Cleanup

Shows mesh, placements, and local slices for each GPU.

Cleans up NCCL communicators.

🧩 Example Output
[Mesh] Initializing mesh with 2 GPUs...
[Mesh] NCCL communicators initialized successfully.
[Mesh] Mesh shape set to [2]
[Mesh] GPU 0 logical coords: [0]
[Mesh] GPU 1 logical coords: [1]
[Mesh] Creating subgroup 'tensor' with devices: 0 1
[DTensor] Global shape: [8 x 4]
[DTensor Test] Printing GPU slices and placements:
[GPU 0] Placement: shard,replicate | Slices per dim: [0,3] [0,3]
[GPU 1] Placement: shard,replicate | Slices per dim: [4,7] [0,3]
[Mesh] Destroyed NCCL communicator.
