#include "dtensor.hpp"
#include <iostream>
#include <cmath>

DTensor::DTensor(const std::vector<int64_t>& shape, Mesh& mesh)
    : shape_(shape), mesh_(mesh) {}

void DTensor::setLayout(const std::vector<std::string>& layout) {
    if (layout.size() != shape_.size())
        throw std::runtime_error("Layout size mismatch with tensor rank");
    layout_ = layout;
}

std::vector<int64_t> DTensor::computeLocalStartEnd(int dim, int coord, int mesh_dim_size) const {
    // Divide tensor dim among mesh_dim_size GPUs if sharded
    int64_t step = shape_[dim] / mesh_dim_size;
    int64_t start = coord * step;
    int64_t end = (coord == mesh_dim_size - 1) ? shape_[dim] : start + step;
    return {start, end};
}

void DTensor::placeData(const float* host_data) {
    slices_.clear();

    const auto& mesh_shape = mesh_.meshShape();
    const auto& mesh_coords = mesh_.meshCoords();
    int num_gpus = mesh_.size();

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        const std::vector<int>& coords = mesh_coords.at(gpu);
        std::vector<std::pair<int,int>> gpu_slices;

        for (size_t dim = 0; dim < shape_.size(); ++dim) {
            if (layout_[dim] == "shard") {
                int mesh_dim_size = (dim < mesh_shape.size()) ? mesh_shape[dim] : 1;
                auto bounds = computeLocalStartEnd(dim, coords[dim], mesh_dim_size);
                gpu_slices.push_back({(int)bounds[0], (int)bounds[1]});
            } 
            else if (layout_[dim] == "replicate") {
                gpu_slices.push_back({0, (int)shape_[dim]});
            }
            else if (layout_[dim] == "partial") {
                gpu_slices.push_back({0, (int)shape_[dim]});
            }
            else {
                throw std::runtime_error("Unknown layout type: " + layout_[dim]);
            }
        }
        slices_[gpu] = gpu_slices;
    }
}

void DTensor::printHostTensor() const {
    std::cout << "[DTensor] Global Tensor shape: [";
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
