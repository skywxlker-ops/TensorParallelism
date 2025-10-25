#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <mutex>

// ---------------- Work ----------------
class Work {
public:
    explicit Work(cudaStream_t stream);
    ~Work();

    void markCompleted(bool success = true);
    bool wait();

private:
    cudaStream_t stream_;
    cudaEvent_t event_;
    bool completed_;
    bool success_;
};

// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
    struct Meta {
        int rank = 0;
        int world_size = 0;
        ncclComm_t comm = nullptr;
    };

    ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
    ~ProcessGroup();

    template<typename T>
    std::shared_ptr<Work> all_reduce(T* data, size_t count, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> reduce_scatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> all_gather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype);

    template<typename T>
    std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype);

    int rank() const;
    int world_size() const;

private:
    void checkCuda(cudaError_t err);
    void ncclCheck(ncclResult_t res);

private:
    int rank_;
    int world_size_;
    int device_;
    Meta meta_;

    cudaStream_t compute_stream_;
    cudaStream_t reduce_scatter_stream_;
    cudaStream_t all_gather_stream_;
    cudaStream_t broadcast_stream_;
    cudaStream_t all_reduce_stream_;
};
