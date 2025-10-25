#include "mesh.hpp"

void Mesh::printMesh() const {
    std::cout << "[Mesh] num_gpus: " << num_gpus_ 
              << ", rows: " << rows_ 
              << ", cols: " << cols_ << std::endl;

    for (int i = 0; i < num_gpus_; ++i) {
        std::cout << "  GPU " << i << " logical coords: [" 
                  << logical_coords_[i] << "]" << std::endl;
    }
}
