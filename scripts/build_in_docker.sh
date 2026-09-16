#!/usr/bin/env bash
# Configure + build ONNXim inside the container.
# Intended to be run via scripts/devshell.sh:
#
#   ./scripts/devshell.sh ./scripts/build_in_docker.sh
#
# The build directory must be named "build": CMakeLists.txt line 6 hardcodes
# include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake").
set -euo pipefail

cd /workspace/ONNXim
mkdir -p build
cd build

conan install .. --build=missing
cmake ..
make -j"$(nproc)"

echo
echo "Built: /workspace/ONNXim/build/bin/Simulator"
