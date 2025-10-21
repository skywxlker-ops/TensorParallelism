#include "placement.hpp"

Placement::Placement(const std::vector<std::string>& layout)
    : layout_(layout) {}

const std::vector<std::string>& Placement::getLayout() const {
    return layout_;
}
