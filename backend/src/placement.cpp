#include "placement.hpp"

Placement::Placement(std::string type, Mesh mesh) : type_(std::move(type)), mesh_(mesh) {}

void Placement::describe() const {
    std::cout << "[Placement] Type: " << type_ << std::endl;
    mesh_.printMesh();
}
