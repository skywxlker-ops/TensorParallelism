#pragma once
#include "mesh.hpp"
#include "process_group.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

class DTensor {
public:
    DTensor(std::vector<int64_t> shape, Mesh* mesh) 
        : shape_(shape), mesh_(mesh) 
    {
        int total_slices = mesh_->numGPUs();
        slices_.resize(total_slices, nullptr);

        for (int i = 0; i < total_slices; ++i) {
            int dev = mesh_->getDeviceId(i);
            cudaSetDevice(dev);
            int size = 1;
            for (auto s : shape_) size *= s / total_slices;
            cudaMalloc(&slices_[i], size * sizeof(float));
        }
    }

    ~DTensor() {
        for (auto ptr : slices_) cudaFree(ptr);
    }

    void fillWithRank() {
        for (size_t i = 0; i < slices_.size(); ++i) {
            int dev = mesh_->getDeviceId(i);
            cudaSetDevice(dev);
            std::vector<float> tmp(4, float(i + 1)); // simple fill for test
            cudaMemcpy(slices_[i], tmp.data(), tmp.size()*sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void printSlices() {
        for (size_t i = 0; i < slices_.size(); ++i) {
            int dev = mesh_->getDeviceId(i);
            cudaSetDevice(dev);
            std::vector<float> tmp(4);
            cudaMemcpy(tmp.data(), slices_[i], tmp.size()*sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Slice " << i << " on device " << dev << ": ";
            for (auto v : tmp) std::cout << v << " ";
            std::cout << "\n";
        }
    }

    std::vector<float*>& slices() { return slices_; }

private:
    Mesh* mesh_;
    std::vector<int64_t> shape_;
    std::vector<float*> slices_;
};
