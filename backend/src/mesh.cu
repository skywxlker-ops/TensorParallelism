#include "mesh.hpp"

Mesh::Mesh(int numPhysicalGPUs) : numGPUs_(numPhysicalGPUs) {
    std::cout << "[Mesh] Initializing mesh with " << numGPUs_ << " GPUs..." << std::endl;
    // logical GPU mapping (for testing)
    logicalToPhysical_.resize(numGPUs_);
    for (int i=0; i<numGPUs_; i++)
        logicalToPhysical_[i] = i;
}

void Mesh::printInfo() const {
    for (int i=0; i<numGPUs_; i++) {
        std::cout << "[Mesh] GPU " << i << " logical coords: [" << i << "]\n";
    }
}
