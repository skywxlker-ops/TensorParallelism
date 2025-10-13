#pragma once
#include "mesh.hpp"
#include <vector>
#include <string>
#include <map>
#include <utility>

class DTensor {
public:
    DTensor(const std::vector<int64_t>& shape, Mesh& mesh);

    void setLayout(const std::vector<std::string>& layout);
    void placeData(const float* host_data);

    void printHostTensor() const; // prints tensor shape
    void printSlices() const;     // prints GPU slices and placements

private:
    std::vector<int64_t> shape_;
    std::vector<std::string> layout_;
    Mesh& mesh_;

    // GPU -> vector of slices per dimension: [start,end)
    std::map<int, std::vector<std::pair<int64_t,int64_t>>> slices_;
};
