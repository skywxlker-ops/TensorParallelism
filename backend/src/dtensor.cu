#include "dtensor.hpp"
#include <cuda_runtime.h>

__global__ void initTensorKernel(float* data, float value, int64_t n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int64_t i = idx; i < n; i += stride){
        data[i] = value;
    }
}

void DTensor::initOnGPU(float* d_data, float value, int64_t n){
    int blockSize = 256;
    int gridSize = (n + blockSize - 1)/blockSize;
    initTensorKernel<<<gridSize, blockSize>>>(d_data, value, n);
    cudaDeviceSynchronize();
}

DTensor::DTensor(const std::vector<int64_t>& shape, Mesh& mesh)
    : shape_(shape), mesh_(mesh) {}

void DTensor::setLayout(const std::vector<std::string>& layout){
    if(layout.size() != shape_.size())
        throw std::runtime_error("Layout size mismatch");
    layout_ = layout;
    computeSlices();
}

void DTensor::computeSlices(){
    slices_.clear();
    int num_gpus = mesh_.size();

    for(int gpu=0;gpu<num_gpus;++gpu){
        std::vector<std::pair<int64_t,int64_t>> gpu_slices;
        auto coords = mesh_.meshCoords().at(gpu);

        for(size_t dim=0;dim<shape_.size();++dim){
            const auto& lay = layout_[dim];
            int64_t step = shape_[dim] / num_gpus;
            int64_t start = coords[0] * step;
            int64_t end = (gpu==num_gpus-1) ? shape_[dim] : start+step;

            gpu_slices.push_back({start,end});
        }
        slices_[gpu] = gpu_slices;
    }
}

void DTensor::placeData(const float* host_data){
    computeSlices();
}

void DTensor::printHostTensor() const{
    std::cout << "[DTensor] Original Host Tensor shape: [";
    for(size_t i=0;i<shape_.size();++i)
        std::cout << shape_[i] << (i+1<shape_.size()?",":"");
    std::cout << "]" << std::endl;
}

void DTensor::printSlices() const{
    for(auto& [gpu, slice_vec]: slices_){
        std::cout << "[GPU " << gpu << "] Placement: ";
        for(size_t dim=0;dim<layout_.size();++dim)
            std::cout << layout_[dim] << (dim+1<layout_.size()?",":"");
        std::cout << " | Slices per dim: ";
        for(auto& s: slice_vec)
            std::cout << "[" << s.first << "," << s.second-1 << "] ";
        std::cout << std::endl;
    }
}
