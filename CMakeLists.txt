cmake_minimum_required(VERSION 3.12)
project(nekos)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Find OpenMP package (optional for parallel graph processing)
# Help CMake find libomp on macOS (installed via Homebrew)
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP)

# Add the library directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Set compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Define whether OpenMP is available
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel processing enabled")
    add_compile_definitions(NEKOS_HAS_OPENMP)
else()
    message(WARNING "OpenMP not found - falling back to serial processing")
endif()

# Option to build Python module
option(BUILD_PYTHON_MODULE "Build Python module" OFF)

# ============================================================================
# Python Module (only if BUILD_PYTHON_MODULE is ON)
# ============================================================================
if(BUILD_PYTHON_MODULE)
    # Find Python and pybind11
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    # Create Python module
    pybind11_add_module(nekos_py
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/python_bindings.cpp
    )

    # Link OpenMP if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nekos_py PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Set module name to '_core' to avoid naming conflicts
    set_target_properties(nekos_py PROPERTIES OUTPUT_NAME _core)

    # Set output directory (use CMAKE_LIBRARY_OUTPUT_DIRECTORY from setup.py if provided)
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(nekos_py PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        )
    endif()

    # Installation
    install(TARGETS nekos_py
        LIBRARY DESTINATION .
    )
endif()
