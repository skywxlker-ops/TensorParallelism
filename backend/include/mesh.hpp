#pragma once
#include <vector>
#include <cuda_runtime.h>

class Mesh {
public:
    Mesh(int numPhysical = 1, int logicalsPerPhysical = 2, int bufSize = 5);
    ~Mesh();

    int getSize() const { return total_logical_; }
    int getTotalLogical() const { return total_logical_; }
    int getBufferSize() const { return buffer_size_; }
    float* getBuffer(int idx) { return buffers_[idx]; }

    void simulateAllReduce(); // sums all logical GPU buffers

private:
    int num_physical_;
    int logicals_per_phys_;
    int total_logical_;
    int buffer_size_;
    std::vector<float*> buffers_;
};
