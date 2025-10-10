#include "dtensor.hpp"

DTensor::DTensor(const std::vector<int64_t>& shape, Mesh& mesh)
    : shape_(shape), mesh_(mesh) {
    size_ = 1;
    for (auto s : shape_) size_ *= s;
    cudaMalloc(&data_, size_ * sizeof(float));
}

DTensor::~DTensor() {
    if (data_) cudaFree(data_);
}

void DTensor::setLayout(const std::vector<std::string>& layout) {
    if (layout.size() != shape_.size())
        throw std::runtime_error("Layout size must match tensor dimensions.");
    layout_ = layout;
}

std::pair<int64_t,int64_t> DTensor::getSliceForDim(int dim, int gpu_id) const {
    const auto& coords_map = mesh_.meshCoords();
    const auto& mesh_shape = mesh_.meshShape();

    auto coords = coords_map.at(gpu_id);

    if (layout_[dim] == "replicate") {
        return {0, shape_[dim]};
    } else if (layout_[dim] == "shard") {
        int64_t chunk_size = shape_[dim] / mesh_shape[dim];
        int64_t start = coords[dim] * chunk_size;
        int64_t end = start + chunk_size;
        return {start, end};
    } else if (layout_[dim] == "partial") {
        return {0, shape_[dim]};
    }
    throw std::runtime_error("Unknown layout type");
}

void DTensor::printSlices() const {
    int num_gpus = mesh_.size();
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        std::cout << "[GPU " << gpu << "] Slices per dim: ";
        for (int d = 0; d < shape_.size(); ++d) {
            auto s = getSliceForDim(d, gpu);
            std::cout << "[" << s.first << "," << s.second << "] ";
        }
        std::cout << std::endl;
    }
}
