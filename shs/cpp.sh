#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../src/cpp"

if [ ! -d 3rdparty/pybind11/pybind11-src ]; then
    git clone --depth 1 --branch v2.12.0 https://github.com/pybind/pybind11.git 3rdparty/pybind11/pybind11-src
fi

PYTHON_BIN="$(command -v python || command -v python3)"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
    -DPython_EXECUTABLE="$PYTHON_BIN"
cmake --build build --config Release -j"$(nproc)"
cp -v ../models/MLCodec_*.so ../entropy_models/
