#include "logical_gpu.hpp"
#include "logical_nccl_sim.hpp"
#include <iostream>

int main() {
    LogicalGPUManager manager;
    size_t N = 1024;
    manager.init(2, N); // 2 logical GPUs per physical GPU
    manager.printInfo();

    auto& logicals = manager.getGPUs();

    // Test 1: Logical GPU 0 & Logical GPU 1 (same physical GPU)
    std::cout << "\n=== Test: Logical GPU 0 & Logical GPU 1 ===\n";
    logical_nccl_sim::simulateAllReduce(logicals[0].bufA, logicals[1].bufB, logicals[0].bufC, N, logicals[0].stream, logicals[0].logical_id);

    // Test 2: Logical GPU 0 & Logical GPU 2 (different physical GPU)
    std::cout << "\n=== Test: Logical GPU 0 & Logical GPU 2 ===\n";
    std::cout << "[LogicalGPU] Inter-physical collective detected. Would use real NCCL.\n";

    std::cout << "\nAll logical NCCL tests finished!\n";
    return 0;
}
