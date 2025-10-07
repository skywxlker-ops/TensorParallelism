#include "mesh.hpp"
#include "task.hpp"

int main() {
    Mesh mesh(2);
    Task::runAllReduce(mesh);
    return 0;
}
