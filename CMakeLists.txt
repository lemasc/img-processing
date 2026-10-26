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

# Smoke test: confirms the CUDA toolchain, GPU, and timing harness work.
add_executable(smoke src/smoke.cu)
target_link_libraries(smoke PRIVATE common)
