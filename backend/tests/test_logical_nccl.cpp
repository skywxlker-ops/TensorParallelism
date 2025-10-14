#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    int num_gpus = 1; // simulate same physical GPU
    int ranks = 2;    // two logical GPUs on same device
    int* d_data[2];

    cudaSetDevice(0);

    // Allocate memory for both "logical GPUs" on same physical GPU
    cudaMalloc(&d_data[0], 128 * sizeof(int));
    cudaMalloc(&d_data[1], 128 * sizeof(int));

    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    ncclComm_t comms[2];

    std::cout << "Attempting NCCL init for two ranks on the same GPU...\n";

    // This will fail!
    NCCL_CHECK(ncclCommInitRank(&comms[0], ranks, id, 0)); // rank 0
    NCCL_CHECK(ncclCommInitRank(&comms[1], ranks, id, 1)); // rank 1 (same GPU!)

    std::cout << "This line likely won't be reached!\n";

    return 0;
}
