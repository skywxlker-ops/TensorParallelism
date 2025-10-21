#include "dtensor.hpp"

DTensor::DTensor(Mesh* mesh, size_t totalSize)
    : mesh_(mesh), totalSize_(totalSize) {}

void DTensor::setLayout(const std::vector<std::string>& layout) {
    layout_ = layout;
}

void DTensor::placeData(const float* host_data) {
    int numGPUs = mesh_->size();
    slicesPerGPU_.resize(numGPUs, std::vector<int>(4,0)); // [row_start,row_end,col_start,col_end]

    if (layout_[0] == "shard") {
        int rowsPerGPU = 8 / numGPUs; // assuming 8 rows for simplicity
        for (int i=0; i<numGPUs; i++) {
            slicesPerGPU_[i][0] = i*rowsPerGPU;
            slicesPerGPU_[i][1] = (i+1)*rowsPerGPU - 1;
        }
    } else { // replicate
        for (int i=0; i<numGPUs; i++) {
            slicesPerGPU_[i][0] = 0;
            slicesPerGPU_[i][1] = 7;
        }
    }

    if (layout_[1] == "shard") {
        int colsPerGPU = 4 / numGPUs; // assuming 4 columns
        for (int i=0; i<numGPUs; i++) {
            slicesPerGPU_[i][2] = i*colsPerGPU;
            slicesPerGPU_[i][3] = (i+1)*colsPerGPU - 1;
        }
    } else { // replicate
        for (int i=0; i<numGPUs; i++) {
            slicesPerGPU_[i][2] = 0;
            slicesPerGPU_[i][3] = 3;
        }
    }
}

void DTensor::printSlices() const {
    for (int i=0; i<mesh_->size(); i++) {
        std::cout << "[GPU " << i << "] row: ["
                  << slicesPerGPU_[i][0] << "," << slicesPerGPU_[i][1] << "], col: ["
                  << slicesPerGPU_[i][2] << "," << slicesPerGPU_[i][3] << "]\n";
    }
}
