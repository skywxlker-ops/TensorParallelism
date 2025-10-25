#include "process_group.hpp"
#include <iostream>
#include <stdexcept>

// ---------------- Work ----------------
Work::Work(cudaStream_t stream) : stream_(stream), completed_(false), success_(true) {
    if (cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) != cudaSuccess)
        throw std::runtime_error("Failed to create CUDA event");
}

Work::~Work() {
    cudaEventDestroy(event_);
}

void Work::markCompleted(bool success) {
    success_ = success;
    completed_ = true;
    cudaEventRecord(event_, stream_);
}

bool Work::wait() {
    if (!completed_) return false;
    cudaEventSynchronize(event_);
    return success_;
}

// ---------------- ProcessGroup ----------------
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
    : rank_(rank), world_size_(world_size), device_(device) 
{
    cudaSetDevice(device_);

    checkCuda(cudaStreamCreate(&compute_stream_));
    checkCuda(cudaStreamCreate(&reduce_scatter_stream_));
    checkCuda(cudaStreamCreate(&all_gather_stream_));
    checkCuda(cudaStreamCreate(&broadcast_stream_));
    checkCuda(cudaStreamCreate(&all_reduce_stream_));

    ncclResult_t res = ncclCommInitRank(&meta_.comm, world_size_, id, rank_);
    if (res != ncclSuccess)
        throw std::runtime_error(std::string("ncclCommInitRank failed: ") + ncclGetErrorString(res));
}

ProcessGroup::~ProcessGroup() {
    ncclCommDestroy(meta_.comm);
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(reduce_scatter_stream_);
    cudaStreamDestroy(all_gather_stream_);
    cudaStreamDestroy(broadcast_stream_);
    cudaStreamDestroy(all_reduce_stream_);
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::all_reduce(T* data, size_t count, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(all_reduce_stream_);
    ncclCheck(ncclAllReduce(data, data, count, dtype, ncclSum, meta_.comm, all_reduce_stream_));
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::reduce_scatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(reduce_scatter_stream_);
    ncclCheck(ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, meta_.comm, reduce_scatter_stream_));
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::all_gather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(all_gather_stream_);
    ncclCheck(ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, meta_.comm, all_gather_stream_));
    work->markCompleted(true);
    return work;
}

template<typename T>
std::shared_ptr<Work> ProcessGroup::broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto work = std::make_shared<Work>(broadcast_stream_);
    ncclCheck(ncclBroadcast(data, data, count, dtype, root, meta_.comm, broadcast_stream_));
    work->markCompleted(true);
    return work;
}

int ProcessGroup::rank() const { return rank_; }
int ProcessGroup::world_size() const { return world_size_; }

void ProcessGroup::checkCuda(cudaError_t err) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
}

void ProcessGroup::ncclCheck(ncclResult_t res) {
    if (res != ncclSuccess)
        throw std::runtime_error(std::string("NCCL Error: ") + ncclGetErrorString(res));
}

// Explicit template instantiations for float and int
template std::shared_ptr<Work> ProcessGroup::all_reduce<float>(float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::reduce_scatter<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::all_gather<float>(float*, float*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::broadcast<float>(float*, size_t, int, ncclDataType_t);

template std::shared_ptr<Work> ProcessGroup::all_reduce<int>(int*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::reduce_scatter<int>(int*, int*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::all_gather<int>(int*, int*, size_t, ncclDataType_t);
template std::shared_ptr<Work> ProcessGroup::broadcast<int>(int*, size_t, int, ncclDataType_t);
