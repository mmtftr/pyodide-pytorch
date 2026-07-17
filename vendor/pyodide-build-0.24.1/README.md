# Pyodide 0.24.1 CMake toolchain

`Emscripten.cmake` is copied without modification from
[`pyodide/pyodide@0.24.1`](https://github.com/pyodide/pyodide/blob/0.24.1/pyodide-build/pyodide_build/tools/cmake/Modules/Platform/Emscripten.cmake).
The published `pyodide-build==0.24.1` wheel references this file but does not
contain it. `scripts/repair_pyodide_build.py` verifies the pinned SHA-256 before
placing it at the package path expected by that release.

The file is distributed under Pyodide's MPL-2.0 license. See the
[upstream license](https://github.com/pyodide/pyodide/blob/0.24.1/LICENSE).
