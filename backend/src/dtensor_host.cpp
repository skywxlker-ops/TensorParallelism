#include "dtensor.hpp"
#include <cuda_runtime.h>
#include <iostream>

DTensor::DTensor(std::vector<int64_t> shape, Mesh& mesh) : shape(shape), mesh(mesh) {
    numel = 1;
    for (auto s : shape) numel *= s;

    device_buffers.resize(mesh.num_gpus);
    for (int i = 0; i < mesh.num_gpus; ++i) {
        cudaSetDevice(mesh.device_ids[i]);
        cudaMalloc(&device_buffers[i], sizeof(float) * numel / mesh.num_gpus);
    }
}

void DTensor::setLayout(std::vector<std::string> layout) {
    this->layout = layout;
    std::cout << "[DTensor] Layout set to {";
    for (auto& l : layout) std::cout << l << " ";
    std::cout << "}\n";
}

void DTensor::printHostTensor() const {
    std::cout << "[DTensor] Host tensor shape: (";
    for (size_t i = 0; i < shape.size(); ++i)
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    std::cout << ")\n";
}

void DTensor::printSlices() const {
    std::cout << "[DTensor] Printing slice distribution across GPUs:\n";
    for (int i = 0; i < mesh.num_gpus; ++i) {
        std::cout << "  GPU " << i << " -> ";
        if (layout.size() > 0)
            std::cout << layout[0] << " ";
        if (layout.size() > 1)
            std::cout << layout[1];
        std::cout << " slice\n";
    }
}
