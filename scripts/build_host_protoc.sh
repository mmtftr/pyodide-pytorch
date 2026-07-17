#!/usr/bin/env bash
set -Eeuo pipefail

source_dir="${1:?usage: build_host_protoc.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}"
output_dir="${2:?usage: build_host_protoc.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}"
protobuf_source="$source_dir/third_party/protobuf/cmake"
protoc="$output_dir/bin/protoc"

if [[ ! -f "$protobuf_source/CMakeLists.txt" ]]; then
  echo "protobuf submodule is not initialized: $protobuf_source" >&2
  exit 1
fi

cmake \
  -S "$protobuf_source" \
  -B "$output_dir" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$output_dir/bin" \
  -Dprotobuf_BUILD_EXAMPLES=OFF \
  -Dprotobuf_BUILD_SHARED_LIBS=OFF \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dprotobuf_WITH_ZLIB=OFF
cmake --build "$output_dir" --target protoc --parallel 3

if [[ ! -x "$protoc" ]]; then
  echo "host protoc was not produced at $protoc" >&2
  exit 1
fi
"$protoc" --version
