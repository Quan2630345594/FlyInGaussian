cmake_minimum_required(VERSION 3.18)
project(gs_renderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")

add_definitions(-O3 -DWITH_CUDA -DTHRUST_IGNORE_CUB_VERSION_CHECK)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PREFIX_RESULT
)
if(NOT TORCH_PREFIX_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not query the active Python Torch installation")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

# GS-SDF source paths
set(
    GS_SDF_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/GS-SDF"
    CACHE PATH
    "Path to the pinned GS-SDF submodule"
)
set(GSPLAT_CPP_DIR ${GS_SDF_ROOT}/submodules/gsplat_cpp/gsplat_cpp)
set(GSPLAT_DIR ${GS_SDF_ROOT}/submodules/gsplat_cpp/submodules/gsplat)
set(GLM_DIR ${GSPLAT_DIR}/gsplat/cuda/csrc/third_party/glm)

if(NOT EXISTS "${GS_SDF_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "GS-SDF is missing. Run: git submodule update --init --recursive")
endif()

# Build the pinned gsplat implementation with the active Python Torch package.
add_subdirectory(
    ${GS_SDF_ROOT}/submodules/gsplat_cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gsplat_cpp
)

# --- Compile ply_utils for PLY loading ---
add_library(ply_utils_torch STATIC
    ${GS_SDF_ROOT}/include/utils/ply_utils/ply_utils_torch.cpp
)
target_include_directories(ply_utils_torch PUBLIC
    ${GS_SDF_ROOT}/include/utils/ply_utils
    ${GS_SDF_ROOT}/include/utils
)
target_link_libraries(ply_utils_torch ${TORCH_LIBRARIES})

# --- Use Torch-bundled pybind11 ---
add_library(_gs_bridge SHARED bindings.cpp)

target_include_directories(_gs_bridge PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${Python3_INCLUDE_DIRS}
    ${GS_SDF_ROOT}/include
    ${GSPLAT_CPP_DIR}
    ${GSPLAT_DIR}
    ${GLM_DIR}
    ${GS_SDF_ROOT}/include/utils/ply_utils
    ${GS_SDF_ROOT}/include/utils
)

target_compile_definitions(_gs_bridge PRIVATE TORCH_EXTENSION_NAME=_gs_bridge)

find_library(
    TORCH_PYTHON_LIBRARY
    torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
    NO_DEFAULT_PATH
)

target_link_libraries(_gs_bridge PRIVATE
    gsplat_cpp
    ply_utils_torch
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Python3::Python
)

set_target_properties(_gs_bridge PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "$ORIGIN"
)
