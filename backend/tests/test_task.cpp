// #include "mesh.hpp"
// #include "task.hpp"

// int main() {
//     Mesh mesh(2);
//     Task::runAllReduce(mesh);
//     return 0;
// }


#include "mesh.hpp"
#include "task.hpp"
#include <vector>
#include <iostream>

int main() {
    int num_gpus = 2;
    int num_elements = 1024;

    Mesh mesh(num_gpus);

    std::vector<float*> d_data(num_gpus);
    Task::initTensors(d_data, mesh, num_elements);

    Task::runAllReduce(mesh, d_data, num_elements);

    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        cudaFree(d_data[i]);
    }

    return 0;
}


