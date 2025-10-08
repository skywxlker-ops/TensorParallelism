#include "mesh.hpp"
#include "task.hpp"
#include <vector>
#include <iostream>

int main() {
    try {
        Mesh mesh;
        int num_gpus = mesh.size();
        int num_elements = 1024;

        std::vector<float*> d_data;
        Task::initTensors(d_data, mesh, num_elements);
        Task::runAllReduce(mesh, d_data, num_elements);

        for (int i = 0; i < num_gpus; ++i) {
            set_device(i, true);
            cudaFree(d_data[i]);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
