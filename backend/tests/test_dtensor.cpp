#include "../include/mesh.hpp"
#include "../include/dtensor.hpp"
#include "../include/process_group.hpp"
#include <thread>

int main() {
    int world_size = 2;

    Mesh mesh(world_size);
    mesh.printMesh();

    DTensor dt({8,4}, &mesh);
    dt.fillWithRank();
    std::cout << "Before AllReduce:\n";
    dt.printSlices();

    // Create ProcessGroup
    ncclUniqueId id = mesh.getNCCLId();
    std::vector<std::thread> threads;
    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([rank, world_size, &dt, &id](){
            ProcessGroup pg(rank, world_size, rank, id);
            dtensorAllReduce(dt, pg);
        });
    }
    for (auto& t : threads) t.join();

    std::cout << "After AllReduce:\n";
    dt.printSlices();

    return 0;
}
