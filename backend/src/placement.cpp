#include "placement.hpp"
#include <iostream>

void printPlacement(const Placement& p) {
    std::string t = (p.type == PlacementType::Sharded) ? "Sharded" :
                    (p.type == PlacementType::Replicated) ? "Replicated" : "Partial";
    std::cout << "[Placement] " << t << " across devices: ";
    for (auto d : p.device_ids) std::cout << d << " ";
    std::cout << " on dim " << p.dim << std::endl;
}