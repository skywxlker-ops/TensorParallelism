#!/bin/bash
# =====================================================
# TensorParallelism Backend Build Script
# =====================================================

set -e  # Exit on first error

echo " Building TensorParallelism Backend..."

# Paths
INCLUDE_DIR="include"
SRC_DIR="src"
TEST_DIR="tests"
BUILD_DIR="build"

# Compiler commands
GXX="g++-11"           # host C++ compiler
NVCC="nvcc"
CXXFLAGS="-I${INCLUDE_DIR}"
LIBS="-lnccl -lcudart -lpthread"

# Create build directory
mkdir -p $BUILD_DIR

# ===================== Compile sources =====================
echo " Compiling cudafunctions.cpp..."
$GXX -std=c++17 $CXXFLAGS -c ${SRC_DIR}/cudafunctions.cpp -o ${BUILD_DIR}/cudafunctions.o

echo " Compiling mesh.cu..."
$NVCC -ccbin $GXX -std=c++17 $CXXFLAGS -c ${SRC_DIR}/mesh.cu -o ${BUILD_DIR}/mesh.o

echo " Compiling dtensor.cu..."
$NVCC -ccbin $GXX -std=c++17 $CXXFLAGS -c ${SRC_DIR}/dtensor.cu -o ${BUILD_DIR}/dtensor.o

echo " Compiling task.cu..."
$NVCC -ccbin $GXX -std=c++17 $CXXFLAGS -c ${SRC_DIR}/task.cu -o ${BUILD_DIR}/task.o

# ===================== Compile test files =====================
echo " Compiling test_mesh.cpp..."
$GXX -std=c++17 $CXXFLAGS -c ${TEST_DIR}/test_mesh.cpp -o ${BUILD_DIR}/test_mesh.o

echo " Compiling test_task.cpp..."
$GXX -std=c++17 $CXXFLAGS -c ${TEST_DIR}/test_task.cpp -o ${BUILD_DIR}/test_task.o

echo " Compiling test_dtensor.cpp..."
$GXX -std=c++17 $CXXFLAGS -c ${TEST_DIR}/test_dtensor.cpp -o ${BUILD_DIR}/test_dtensor.o

# ===================== Link tests =====================
echo " Linking test_mesh..."
$GXX ${BUILD_DIR}/cudafunctions.o ${BUILD_DIR}/mesh.o ${BUILD_DIR}/test_mesh.o $LIBS -o ${BUILD_DIR}/test_mesh

echo " Linking test_task..."
$GXX ${BUILD_DIR}/cudafunctions.o ${BUILD_DIR}/mesh.o ${BUILD_DIR}/task.o ${BUILD_DIR}/test_task.o $LIBS -o ${BUILD_DIR}/test_task

echo " Linking test_dtensor..."
$GXX ${BUILD_DIR}/cudafunctions.o ${BUILD_DIR}/mesh.o ${BUILD_DIR}/dtensor.o ${BUILD_DIR}/test_dtensor.o $LIBS -o ${BUILD_DIR}/test_dtensor

echo " Build successful!"
echo ""
echo "Run tests with:"
echo "  ./build/test_mesh"
echo "  ./build/test_task"
echo "  ./build/test_dtensor"
