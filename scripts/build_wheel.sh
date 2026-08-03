#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$(cd "${1:?usage: build_wheel.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}" && pwd)"
output_dir="${2:?usage: build_wheel.sh PYTORCH_SOURCE_DIR OUTPUT_DIR}"
max_jobs_override="${PYTORCH_MAX_JOBS_OVERRIDE:-}"

while IFS='=' read -r name value; do
  export "$name=$value"
done < <(python3 "$repo_root/scripts/config.py" env)

if [[ -n "$max_jobs_override" ]]; then
  [[ "$max_jobs_override" =~ ^[1-9][0-9]*$ ]] || {
    echo "PYTORCH_MAX_JOBS_OVERRIDE must be a positive integer" >&2
    exit 1
  }
  export MAX_JOBS="$max_jobs_override"
fi

: "${PYTORCH_HOST_PROTOC:?PYTORCH_HOST_PROTOC must point to the native protoc}"
: "${PYODIDE_LAPACK_LIBRARY:?PYODIDE_LAPACK_LIBRARY must point to libopenblas.so}"
[[ -x "$PYTORCH_HOST_PROTOC" ]] || {
  echo "PYTORCH_HOST_PROTOC is not executable: $PYTORCH_HOST_PROTOC" >&2
  exit 1
}
[[ -f "$PYODIDE_LAPACK_LIBRARY" ]] || {
  echo "Pyodide LAPACK library is missing: $PYODIDE_LAPACK_LIBRARY" >&2
  exit 1
}
PYODIDE_LAPACK_LIBRARY="$(realpath "$PYODIDE_LAPACK_LIBRARY")"
export PYODIDE_LAPACK_LIBRARY
lapack_magic="$(od -An -tx1 -N8 "$PYODIDE_LAPACK_LIBRARY" | tr -d ' \n')"
[[ "$lapack_magic" == "0061736d01000000" ]] || {
  echo "Pyodide LAPACK library is not WebAssembly: $PYODIDE_LAPACK_LIBRARY" >&2
  exit 1
}

actual_ref="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_ref" == "$PYTORCH_REF" ]] || {
  echo "expected PyTorch $PYTORCH_REF, found $actual_ref" >&2
  exit 1
}

# Ccache includes SOURCE_DATE_EPOCH in compiler-result keys because the value
# controls __DATE__/__TIME__ expansion. Derive and export one canonical value
# here so direct local builds, CI builds, and cache probes cannot accidentally
# populate separate namespaces for otherwise identical compiler commands.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$source_dir" show -s --format=%ct HEAD)}"
[[ "$SOURCE_DATE_EPOCH" =~ ^[0-9]+$ ]] || {
  echo "SOURCE_DATE_EPOCH must be a non-negative integer" >&2
  exit 1
}

python3 "$repo_root/scripts/stage_webgpu_sources.py" "$source_dir"

actual_emscripten="$(pyodide config get emscripten_version)"
[[ "$actual_emscripten" == "$EMSCRIPTEN_VERSION" ]] || {
  echo "pyodide-build expects Emscripten $actual_emscripten, manifest pins $EMSCRIPTEN_VERSION" >&2
  exit 1
}
emcc --version | head -1

export PYODIDE_PYTHON_INCLUDE_DIR
PYODIDE_PYTHON_INCLUDE_DIR="$(pyodide config get python_include_dir)"
[[ -f "$PYODIDE_PYTHON_INCLUDE_DIR/Python.h" ]] || {
  echo "Pyodide target Python headers are missing: $PYODIDE_PYTHON_INCLUDE_DIR/Python.h" >&2
  exit 1
}
echo "Pyodide target Python headers: $PYODIDE_PYTHON_INCLUDE_DIR"

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
export CMAKE_C_STANDARD=17
export CMAKE_CXX_STANDARD=20
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

# pyodide-build 0.36.0's vendored CMake toolchain appends SIDE_MODULE_* flags
# each time CMake reads it. CMake reads a toolchain before this hook and may
# read it again while enabling C/CXX, so the bootstrap installs a post-project
# normalizer. Without it, a no-op configure changes every compiler command and
# turns a reusable ccache into a full rebuild.
cmake_project_include_before="$repo_root/scripts/cmake/pyodide_project_include_before.cmake"
[[ -f "$cmake_project_include_before" ]] || {
  echo "CMake side-module bootstrap is missing: $cmake_project_include_before" >&2
  exit 1
}
if [[ -n "${CMAKE_PROJECT_INCLUDE_BEFORE:-}" \
      && "$CMAKE_PROJECT_INCLUDE_BEFORE" != "$cmake_project_include_before" ]]; then
  export PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE="$CMAKE_PROJECT_INCLUDE_BEFORE"
else
  unset PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE || true
fi
if [[ -n "${CMAKE_PROJECT_INCLUDE:-}" ]]; then
  export PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE="$CMAKE_PROJECT_INCLUDE"
else
  unset PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE || true
fi
export CMAKE_PROJECT_INCLUDE_BEFORE="$cmake_project_include_before"

if command -v ccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache

  # pywasmcross is a stable Python wrapper whose adjacent JSON file supplies
  # the actual target compiler flags. Ccache only sees the wrapper command, so
  # hash that hidden input explicitly or a flag/ABI change could reuse a stale
  # object. Content-based compiler identity also keeps regenerated wrapper
  # symlinks from invalidating otherwise identical entries.
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$source_dir}"
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
  export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_ctime,include_file_mtime}"
  pywasmcross_env="$source_dir/.pyodide_build/pywasmcross_symlinks/pywasmcross_env.json"
  case ":${CCACHE_EXTRAFILES:-}:" in
    *":$pywasmcross_env:"*) ;;
    "::") export CCACHE_EXTRAFILES="$pywasmcross_env" ;;
    *) export CCACHE_EXTRAFILES="$CCACHE_EXTRAFILES:$pywasmcross_env" ;;
  esac
fi

# Isolated pyodide-build invocations create a fresh /tmp/build-env-* directory.
# Repair the generated Ninja cache entry before pyodide-build inspects it so a
# vanished environment does not invalidate the entire CMake graph. Isolation
# must remain enabled: Pyodide overlays wasm32-specific files (notably NumPy's
# generated ABI headers) into that environment before invoking PyTorch's build.
cmake_cache="$source_dir/build/CMakeCache.txt"
stable_ninja="$(command -v ninja)"
if [[ -f "$cmake_cache" ]]; then
  test -x "$stable_ninja"
  expected_ninja_cache="CMAKE_MAKE_PROGRAM:FILEPATH=$stable_ninja"
  if ! grep -Fqx "$expected_ninja_cache" "$cmake_cache"; then
    sed -i \
      "s#^CMAKE_MAKE_PROGRAM:FILEPATH=.*#$expected_ninja_cache#" \
      "$cmake_cache"
  fi
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
cd "$source_dir"
pyodide build \
  --skip-emscripten-install \
  --exports=whole_archive \
  --outdir "$output_dir"
