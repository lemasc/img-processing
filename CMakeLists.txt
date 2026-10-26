cmake_minimum_required(VERSION 3.24)
project(hct_imgproc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3060
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(common INTERFACE)
target_include_directories(common INTERFACE include include/third_party)

add_library(image_io STATIC src/image_io.cpp)
target_link_libraries(image_io PUBLIC common)

# Scalar CPU reference (ladder rung 1): auto-vectorization off.
add_library(color_cpu STATIC src/color_cpu.cpp)
target_link_libraries(color_cpu PUBLIC common)
target_compile_options(color_cpu PRIVATE -O2 -fno-tree-vectorize -fno-tree-slp-vectorize)

add_library(color_cuda STATIC src/color_cuda.cu)
target_link_libraries(color_cuda PUBLIC common)

# Smoke test: confirms the CUDA toolchain, GPU, and timing harness work.
add_executable(smoke apps/smoke.cu)
target_link_libraries(smoke PRIVATE common)

# Stage 1: color adjustment.
add_executable(stage1_color apps/stage1_color.cu)
target_link_libraries(stage1_color PRIVATE common image_io color_cpu color_cuda)
