#include "dtensor.hpp"
#include <iostream>
#include <cmath>

DTensor::DTensor(const std::vector<int64_t>& shape, Mesh& mesh)
    : shape_(shape), mesh_(mesh) {}

void DTensor::setLayout(const std::vector<std::string>& layout) {
    if (layout.size() != shape_.size()) throw std::runtime_error("Layout size mismatch");
    layout_ = layout;
}

void DTensor::placeData(const float* host_data) {
    int num_gpus = mesh_.size();
    slices_.clear();

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        std::vector<std::pair<int,int>> gpu_slices;
        for (size_t dim = 0; dim < shape_.size(); ++dim) {
            if (layout_[dim] == "shard") {
                int64_t step = shape_[dim] / num_gpus;
                int64_t start = gpu * step;
                int64_t end = (gpu == num_gpus-1) ? shape_[dim] : start + step;
                gpu_slices.push_back({start,end});
            } else if (layout_[dim] == "replicate") {
                gpu_slices.push_back({0, shape_[dim]});
            } else {
                throw std::runtime_error("Unknown layout type");
            }
        }
        slices_[gpu] = gpu_slices;
    }
}

void DTensor::printHostTensor() const {
    std::cout << "[DTensor] Original Host Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i)
        std::cout << shape_[i] << (i + 1 < shape_.size() ? "," : "");
    std::cout << "]" << std::endl;
}

void DTensor::printSlices() const {
    for (auto& [gpu, slice_vec] : slices_) {
        std::cout << "[GPU " << gpu << "] Placement: ";
        for (size_t dim = 0; dim < layout_.size(); ++dim)
            std::cout << layout_[dim] << (dim + 1 < layout_.size() ? "," : " ");

        std::cout << " | Slices per dim: ";
        for (auto& s : slice_vec)
            std::cout << "[" << s.first << "," << (s.second-1) << "] ";
        std::cout << std::endl;
    }
}

