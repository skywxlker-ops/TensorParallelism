#pragma once
#include "mesh.hpp"
#include <vector>

class Task {
public:
    Task(Mesh& mesh, int elements_per_tensor);
    void run();

private:
    Mesh& mesh_;
    int num_elements_;
    std::vector<float*> buffers_;
};
