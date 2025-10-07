// #pragma once
// #include "mesh.hpp"

// class Task {
// public:
//     static void runAllReduce(Mesh& mesh);
// };


#pragma once
#include "mesh.hpp"
#include <vector>

class Task {
public:
    static void initTensors(std::vector<float*>& d_data, Mesh& mesh, int num_elements = 1024);
    static void runAllReduce(Mesh& mesh, std::vector<float*>& d_data, int num_elements = 1024);
};
