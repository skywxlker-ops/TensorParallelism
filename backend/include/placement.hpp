#pragma once
#include <vector>
#include <string>

enum class PlacementType { Sharded, Replicated, Partial };

struct Placement {
    PlacementType type;
    std::vector<int> device_ids;  // Logical device IDs involved
    std::string dim;              // Dimension along which it's sharded
};