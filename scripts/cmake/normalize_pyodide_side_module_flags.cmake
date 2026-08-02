# pyodide-build 0.36.0 appends SIDE_MODULE_* to CMake's global flags every
# time its toolchain file is read. CMake may read a toolchain more than once,
# so a no-op reconfigure can otherwise change every compile command and defeat
# ccache. pywasmcross already supplies these flags when it invokes emcc/em++, so
# CMake's outer command must contain none of the injected blocks.

if(
  DEFINED PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE
  AND NOT "${PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}" STREQUAL ""
  AND NOT "${PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}" STREQUAL
          "${CMAKE_CURRENT_LIST_FILE}"
)
  include("${PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE}")
endif()

function(
  _pytorch_remove_exact_flag_block
  input_flags
  environment_variable
  output_variable
)
  set(_pytorch_flags "${input_flags}")
  if(
    DEFINED ENV{${environment_variable}}
    AND NOT "$ENV{${environment_variable}}" STREQUAL ""
  )
    # Prefer consuming the single separator added by the toolchain so repeated
    # configure passes do not accumulate harmless-but-different whitespace.
    # The final replacement also handles a block separated by tabs/newlines or
    # supplied as the entire value without rewriting unrelated whitespace.
    string(
      REPLACE " $ENV{${environment_variable}}" ""
      _pytorch_flags "${_pytorch_flags}"
    )
    string(
      REPLACE "$ENV{${environment_variable}} " ""
      _pytorch_flags "${_pytorch_flags}"
    )
    string(
      REPLACE "$ENV{${environment_variable}}" ""
      _pytorch_flags "${_pytorch_flags}"
    )
    string(STRIP "${_pytorch_flags}" _pytorch_flags)
  endif()
  set(${output_variable} "${_pytorch_flags}" PARENT_SCOPE)
endfunction()

macro(_pytorch_strip_side_module_flags variable environment_variable)
  if(DEFINED ${variable})
    _pytorch_remove_exact_flag_block(
      "${${variable}}" "${environment_variable}" _pytorch_live_flags
    )
    set(${variable} "${_pytorch_live_flags}")

    # CMAKE_<LANG>_FLAGS and the non-INIT linker variables normally have cache
    # entries. Strip the cache's own value independently: a nested project may
    # add warnings or other scoped flags to the live variable, and copying that
    # expanded value into the base cache makes every reconfigure grow it.
    get_property(
      _pytorch_has_cache_entry CACHE "${variable}" PROPERTY TYPE SET
    )
    if(_pytorch_has_cache_entry)
      get_property(
        _pytorch_cached_flags CACHE "${variable}" PROPERTY VALUE
      )
      _pytorch_remove_exact_flag_block(
        "${_pytorch_cached_flags}"
        "${environment_variable}"
        _pytorch_clean_cached_flags
      )
      set_property(
        CACHE "${variable}" PROPERTY VALUE "${_pytorch_clean_cached_flags}"
      )
    endif()
  endif()
endmacro()

_pytorch_strip_side_module_flags(CMAKE_C_FLAGS SIDE_MODULE_CFLAGS)
_pytorch_strip_side_module_flags(CMAKE_CXX_FLAGS SIDE_MODULE_CXXFLAGS)
foreach(_pytorch_linker_kind IN ITEMS SHARED MODULE)
  _pytorch_strip_side_module_flags(
    CMAKE_${_pytorch_linker_kind}_LINKER_FLAGS
    SIDE_MODULE_LDFLAGS
  )
  _pytorch_strip_side_module_flags(
    CMAKE_${_pytorch_linker_kind}_LINKER_FLAGS_INIT
    SIDE_MODULE_LDFLAGS
  )
endforeach()

unset(_pytorch_live_flags)
unset(_pytorch_cached_flags)
unset(_pytorch_clean_cached_flags)
unset(_pytorch_has_cache_entry)
unset(_pytorch_linker_kind)
unset(_pytorch_strip_side_module_flags)
