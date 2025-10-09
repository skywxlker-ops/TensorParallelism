#include "mesh.hpp"

Mesh::Mesh() {
    num_gpus_ = device_count_ensure_non_zero();
    std::cout << "[Mesh] Initializing mesh with " << num_gpus_ << " GPUs..." << std::endl;

    comms_.resize(num_gpus_);
    streams_.resize(num_gpus_);

    std::vector<int> devs(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) {
        devs[i] = i;
        DeviceIndex old = ExchangeDevice(i);
        cudaStreamCreate(&streams_[i]);
        ExchangeDevice(old);
    }

    NCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, devs.data()));
    std::cout << "[Mesh] NCCL communicators initialized successfully." << std::endl;
}

Mesh::~Mesh() {
    for (int i = 0; i < num_gpus_; ++i) {
        DeviceIndex old = ExchangeDevice(i);
        ncclCommDestroy(comms_[i]);
        cudaStreamDestroy(streams_[i]);
        ExchangeDevice(old);
    }
    for (auto& [name, comm] : subgroups_) ncclCommDestroy(comm);
    std::cout << "[Mesh] Destroyed NCCL communicator." << std::endl;
}

void Mesh::setMeshShape(const std::vector<int64_t>& shape) {
    mesh_shape_ = shape.empty() ? std::vector<int64_t>{num_gpus_} : shape;

    int total = 1;
    for (auto s : mesh_shape_) total *= s;
    if (total != num_gpus_) throw std::runtime_error("Mesh shape product must equal number of GPUs.");

    mesh_coords_.clear();
    std::vector<int64_t> coords(mesh_shape_.size(), 0);
    for (int i = 0; i < num_gpus_; ++i) {
        mesh_coords_[i] = std::vector<int>(coords.begin(), coords.end());

        // Increment coordinates for next GPU
        for (int d = mesh_shape_.size() - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < mesh_shape_[d]) break;
            coords[d] = 0;
        }
    }

    // Print mesh shape
    std::cout << "[Mesh] Mesh shape set to [";
    for (size_t i = 0; i < mesh_shape_.size(); ++i) {
        std::cout << mesh_shape_[i] << (i + 1 < mesh_shape_.size() ? "x" : "");
    }
    std::cout << "]" << std::endl;

    // Print logical coordinates for each GPU
    for (int i = 0; i < num_gpus_; ++i) {
        std::cout << "[Mesh] GPU " << i << " logical coords: [";
        for (size_t d = 0; d < mesh_coords_[i].size(); ++d) {
            std::cout << mesh_coords_[i][d] << (d + 1 < mesh_coords_[i].size() ? "," : "");
        }
        std::cout << "]" << std::endl;
    }
}


void Mesh::createSubGroup(const std::string& name, const std::vector<int>& devices) {
    std::cout << "[Mesh] Creating subgroup '" << name << "' with devices: ";
    for (auto d : devices) std::cout << d << " ";
    std::cout << std::endl;

    ncclComm_t subcomm;
    NCCL_CHECK(ncclCommInitAll(&subcomm, devices.size(), devices.data()));
    subgroups_[name] = subcomm;
}

ncclComm_t Mesh::getSubComm(const std::string& name) const {
    auto it = subgroups_.find(name);
    if (it == subgroups_.end()) throw std::runtime_error("Subgroup not found: " + name);
    return it->second;
}

void Mesh::allReduce(float* data, int num_elements) const {
    int rank;
    cudaGetDevice(&rank);
    NCCL_CHECK(ncclAllReduce(data, data, num_elements, ncclFloat, ncclSum, comms_[rank], streams_[rank]));
    cudaDeviceSynchronize();
}
