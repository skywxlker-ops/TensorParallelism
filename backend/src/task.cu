#include "task.hpp"
#include <iostream>

Task::Task(Mesh& mesh, int elements_per_tensor)
    : mesh_(mesh), num_elements_(elements_per_tensor) 
{
    buffers_.resize(4); // 4 logical GPUs for this example
    for (int i = 0; i < 4; ++i) {
        buffers_[i] = new float[num_elements_];
        for (int j = 0; j < num_elements_; ++j)
            buffers_[i][j] = static_cast<float>(i + j);
    }
}

void Task::run() {
    std::cout << "[Task] Running AllReduce...\n";
    mesh_.allReduce(buffers_, num_elements_);
    std::cout << "[Task] AllReduce done. Sample values:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "GPU " << i << ": ";
        for (int j = 0; j < std::min(5, num_elements_); ++j)
            std::cout << buffers_[i][j] << " ";
        std::cout << "\n";
    }
}
