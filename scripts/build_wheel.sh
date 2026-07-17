#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$(cd "${1:?usage: build_wheel.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}" && pwd)"
output_dir="${2:?usage: build_wheel.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}"

while IFS='=' read -r name value; do
  export "$name=$value"
done < <(python3 "$repo_root/scripts/config.py" env)

: "${PYTORCH_HOST_PROTOC:?PYTORCH_HOST_PROTOC must point to the native protoc}"
[[ -x "$PYTORCH_HOST_PROTOC" ]] || {
  echo "PYTORCH_HOST_PROTOC is not executable: $PYTORCH_HOST_PROTOC" >&2
  exit 1
}

actual_ref="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_ref" == "$PYTORCH_REF" ]] || {
  echo "expected PyTorch $PYTORCH_REF, found $actual_ref" >&2
  exit 1
}

actual_emscripten="$(pyodide config get emscripten_version)"
[[ "$actual_emscripten" == "$EMSCRIPTEN_VERSION" ]] || {
  echo "pyodide-build expects Emscripten $actual_emscripten, manifest pins $EMSCRIPTEN_VERSION" >&2
  exit 1
}
emcc --version | head -1

export ATEN_THREADING=NATIVE
export BLAS=Eigen
export BUILD_BINARY=0
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0
export BUILD_CUSTOM_PROTOBUF=1
export BUILD_FUNCTORCH=0
export BUILD_LAZY_TS_BACKEND=0
export BUILD_SHARED_LIBS=0
export BUILD_TEST=0
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja
export CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
export CMAKE_POSITION_INDEPENDENT_CODE=ON
export CMAKE_CXX_STANDARD=17
export MAX_JOBS
export PIP_CONSTRAINT="$repo_root/config/build-constraints.txt"
export PYTHONHASHSEED=0
export PYTORCH_BUILD_NUMBER=1
export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
export USE_BLAS=1
export USE_CUDA=0
export USE_DISTRIBUTED=0
export USE_EIGEN_FOR_BLAS=1
export USE_EXCEPTION_PTR=1
export USE_FBGEMM=0
export USE_FFMPEG=0
export USE_GFLAGS=0
export USE_GLOO=0
export USE_GLOG=0
export USE_ITT=0
export USE_KINETO=0
export USE_LAPACK=0
export USE_LEVELDB=0
export USE_LMDB=0
export USE_MKLDNN=0
export USE_MKL=0
export USE_MPI=0
export USE_NCCL=0
export USE_NNPACK=0
export USE_NUMA=0
export USE_NUMPY=1
export USE_OPENCL=0
export USE_OPENCV=0
export USE_OPENMP=0
export USE_PYTORCH_QNNPACK=0
export USE_QNNPACK=0
export USE_ROCM=0
export USE_TBB=0
export USE_TENSORPIPE=0
export USE_TENSORRT=0
export USE_VULKAN=0
export USE_XNNPACK=0
export USE_ZMQ=0
export USE_ZSTD=0

export CXXFLAGS="${CXXFLAGS:-} -fexceptions"
export LDFLAGS="${LDFLAGS:-} -sDISABLE_EXCEPTION_CATCHING=0"

if command -v ccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
cd "$source_dir"
pyodide build --exports=whole_archive --outdir "$output_dir"
