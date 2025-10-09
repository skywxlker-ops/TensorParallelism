#include "mesh.hpp"
#include "task.hpp"
#include <vector>

int main() {
    Mesh mesh;
    mesh.setMeshShape({2}); // auto 1D
    mesh.createSubGroup("tensor", {0,1});

    int num_elements = 1024;
    int num_gpus = mesh.size();
    std::vector<float*> d_data(num_gpus);

    Task::initTensors(d_data, mesh, num_elements);
    Task::runAllReduce(mesh, d_data, num_elements);

    for (int i = 0; i < num_gpus; ++i) {
        DeviceIndex old = ExchangeDevice(i);
        cudaFree(d_data[i]);
        ExchangeDevice(old);
    }

    return 0;
}
