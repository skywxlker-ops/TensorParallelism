#pragma once
#include <vector>
#include <string>

class Placement {
public:
    // layout = {"shard", "replicate"} per dimension
    Placement(const std::vector<std::string>& layout);
    const std::vector<std::string>& getLayout() const;

private:
    std::vector<std::string> layout_;
};
