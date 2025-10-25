#include "dtensor.hpp"
#include "process_group.hpp"

void dtensorAllReduce(DTensor& dt, ProcessGroup& pg) {
    for (auto& slice : dt.slices()) {
        pg.all_reduce(slice, 4, ncclFloat32)->wait(); // assuming slice size = 4 for demo
    }
}
