#pragma once
#include "mesh.hpp"
#include <vector>

void runAllReduceTask(Mesh& mesh, std::vector<float*>& buffers, int N);
