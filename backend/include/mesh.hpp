#pragma once
#include <vector>
#include <iostream>

class Mesh {
public:
    Mesh(int num_logical = 4, int buffer_size = 5);
    ~Mesh();

    int getSize() const { return total_logical_; }
    float* getBuffer(int idx) { return buffers_[idx].data(); }

    void simulateAllReduce();

private:
    int total_logical_;
    int buffer_size_;
    std::vector<std::vector<float>> buffers_;
};
