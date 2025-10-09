#include "mesh.hpp"

int main() {
    Mesh mesh;
    mesh.setMeshShape({2}); // auto 1D, can do {2,2} for 4 GPUs
    mesh.createSubGroup("tensor", {0,1});
    return 0;
}
