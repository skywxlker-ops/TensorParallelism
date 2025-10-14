#include "../include/logical_gpu.hpp"

int main() {
    LogicalGPUManager manager;

    // Example: 2 physical GPUs, 2 logical GPUs each, 1024 memory per logical GPU
    manager.createLogicalGPUs(2, 2, 1024);

    std::cout << "Total Logical GPUs: " << manager.totalLogicalGPUs() << std::endl;
    manager.printLogicalGPUs();

    return 0;
}
