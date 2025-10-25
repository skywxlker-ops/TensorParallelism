#pragma once
#include <string>
#include <iostream>
#include "mesh.hpp"
#include "process_group.hpp"

class Placement {
public:
    Placement(std::string type, Mesh mesh);
    void describe() const;
private:
    std::string type_;
    Mesh mesh_;
};
