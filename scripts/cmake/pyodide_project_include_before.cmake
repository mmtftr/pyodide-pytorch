# pyodide-build's CMake toolchain exposes CMAKE_PROJECT_INCLUDE_BEFORE, but
# that hook runs before CMake may reread the toolchain while enabling C and
# CXX. Register the real normalizer as the matching post-project hook instead.

set(
  _pytorch_side_module_normalizer
  "${CMAKE_CURRENT_LIST_DIR}/normalize_pyodide_side_module_flags.cmake"
)

# CMake 3.27 also accepts only one CMAKE_PROJECT_INCLUDE_BEFORE file. Run the
# caller's original bootstrap first so any variables it establishes remain
# visible to both this bootstrap and the rest of project().
if(
  DEFINED ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE}
  AND NOT "$ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE}" STREQUAL ""
  AND NOT "$ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE}" STREQUAL
          "${CMAKE_CURRENT_LIST_FILE}"
)
  include("$ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE}")
endif()

# CMake 3.27 accepts only one CMAKE_PROJECT_INCLUDE file. Preserve a caller's
# hook and let the normalizer invoke it before removing injected flags. The
# environment fallback is populated by scripts/build_wheel.sh when it replaces
# an existing CMAKE_PROJECT_INCLUDE_BEFORE value with this bootstrap.
if(
  DEFINED CMAKE_PROJECT_INCLUDE
  AND NOT "${CMAKE_PROJECT_INCLUDE}" STREQUAL ""
  AND NOT "${CMAKE_PROJECT_INCLUDE}" STREQUAL
          "${_pytorch_side_module_normalizer}"
)
  set(
    PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE
    "${CMAKE_PROJECT_INCLUDE}"
  )
elseif(
  DEFINED ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}
  AND NOT "$ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}" STREQUAL ""
)
  set(
    PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE
    "$ENV{PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}"
  )
endif()

set(CMAKE_PROJECT_INCLUDE "${_pytorch_side_module_normalizer}")
unset(_pytorch_side_module_normalizer)
