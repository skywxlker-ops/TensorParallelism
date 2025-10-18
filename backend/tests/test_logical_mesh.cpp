#include "task.hpp"
#include "mesh.hpp"

int main() {
    Mesh mesh(4, 5); // 4 logical GPUs, buffer of size 5
    runAllReduceTask(mesh);
    std::cout << "Test completed.\n";
    return 0;
}
