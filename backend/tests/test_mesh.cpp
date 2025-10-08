#include "mesh.hpp"

int main() {
    try {
        Mesh mesh;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
