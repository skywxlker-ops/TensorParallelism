#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "cudafunctions.hpp"

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class Mesh {
public:
    Mesh();
    ~Mesh();

    int size() const { return num_gpus_; }
    ncclComm_t getComm(int rank) const { return comms_.at(rank); }
    cudaStream_t getStream(int rank) const { return streams_.at(rank); }

    // Topology / Layout
    void setMeshShape(const std::vector<int64_t>& shape = {});
    const std::vector<int64_t>& meshShape() const { return mesh_shape_; }
    const std::map<int, std::vector<int>>& meshCoords() const { return mesh_coords_; }

    // Subgroups
    void createSubGroup(const std::string& name, const std::vector<int>& devices);
    ncclComm_t getSubComm(const std::string& name) const;

    // NCCL helpers
    void allReduce(float* data, int num_elements) const;

private:
    int num_gpus_;
    std::vector<int64_t> mesh_shape_;
    std::map<int, std::vector<int>> mesh_coords_;

    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
    std::map<std::string, ncclComm_t> subgroups_;
};
