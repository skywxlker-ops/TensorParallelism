#!/bin/bash
# =====================================================
# TensorParallelism Backend Build Script
# =====================================================

set -e  # Exit on first error

echo "🚀 Building TensorParallelism Backend..."

# Paths
INCLUDE_DIR="include"
SRC_DIR="src"
TEST_DIR="tests"
BUILD_DIR="build"

# Create build directory
mkdir -p $BUILD_DIR

# Compiler flags
CXXFLAGS="-I${INCLUDE_DIR} -lnccl"
NVCC="nvcc"

# =====================================================
# Compile mesh and task tests
# =====================================================
echo "🧩 Compiling test_mesh..."
$NVCC $CXXFLAGS -o ${BUILD_DIR}/test_mesh \
    ${TEST_DIR}/test_mesh.cpp \
    ${SRC_DIR}/mesh.cu \
    ${SRC_DIR}/cudafunctions.cpp

echo "🧠 Compiling test_task..."
$NVCC $CXXFLAGS -o ${BUILD_DIR}/test_task \
    ${TEST_DIR}/test_task.cpp \
    ${SRC_DIR}/mesh.cu \
    ${SRC_DIR}/task.cu \
    ${SRC_DIR}/cudafunctions.cpp

echo "✅ Build successful!"
echo ""
echo "Run tests with:"
echo "  ./build/test_mesh"
echo "  ./build/test_task"
