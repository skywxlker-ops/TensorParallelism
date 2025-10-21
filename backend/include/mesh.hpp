#pragma once
#include <vector>
#include <iostream>

class Mesh {
public:
    Mesh(int numPhysicalGPUs);

    int size() const { return numGPUs_; }
    void printInfo() const;

private:
    int numGPUs_;
    std::vector<int> logicalToPhysical_; // for logical GPU testing
};
