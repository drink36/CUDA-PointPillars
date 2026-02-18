cmake_minimum_required(VERSION 3.18)
project(pointpillar LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# -------- CUDA arch (optional) --------
if(DEFINED ENV{CUDASM} AND NOT "$ENV{CUDASM}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDASM})
endif()

# -------- TensorRT (keep the exact working lib dir logic if you want) --------
# TensorRT
set(TRT_LIB_DIR /usr/local/cuda/targets/aarch64-linux/lib)  # 跟能動那份一致

find_library(NVINFER_LIB NAMES nvinfer HINTS ${TRT_LIB_DIR} REQUIRED)
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    "$ENV{TensorRT_Inc}"
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/local/include
  REQUIRED
)

find_library(NVINFER_LIB
  NAMES nvinfer
  HINTS ${TRT_LIB_DIR}
  REQUIRED
)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "NVINFER: ${NVINFER_LIB}")

# -------- Sources (match your old layout) --------
file(GLOB_RECURSE CORE_FILES CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/pointpillar/*.cu
  ${PROJECT_SOURCE_DIR}/src/pointpillar/*.cpp
  ${PROJECT_SOURCE_DIR}/src/common/tensor.cu
  ${PROJECT_SOURCE_DIR}/src/common/tensorrt.cpp
)

# -------- libpointpillar_core.so --------
add_library(pointpillar_core SHARED ${CORE_FILES})

target_include_directories(pointpillar_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/pointpillar
    ${TENSORRT_INCLUDE_DIR}
    
)

target_link_libraries(pointpillar_core
  PRIVATE
    ${NVINFER_LIB}
    CUDA::cudart

)

# -------- pointpillar executable --------
add_executable(pointpillar ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(pointpillar
  PRIVATE
    pointpillar_core
    CUDA::cudart

)

set_target_properties(pointpillar PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
