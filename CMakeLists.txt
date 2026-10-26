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
# -ffinite-math-only lets GCC emit minss/maxss for the clamp. Without it the clamp compiles to
# branches that mispredict on noisy images (see experiments/clamp_branches.cpp).
target_compile_options(color_cpu PRIVATE -O2 -fno-tree-vectorize -fno-tree-slp-vectorize
                       -ffinite-math-only -fno-signed-zeros)

add_library(color_cuda STATIC src/color_cuda.cu)
target_link_libraries(color_cuda PUBLIC common)
# nvcc fuses multiply-adds by default, the CPU baseline does not. OFF gives the unfused GPU result
# (cmake -DCOLOR_CUDA_FMAD=OFF), used to explain 1-level differences in Stage 3.
option(COLOR_CUDA_FMAD "Allow nvcc to fuse multiply-add in color_cuda" ON)
if(NOT COLOR_CUDA_FMAD)
  target_compile_options(color_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

add_library(compare STATIC src/compare.cpp)
target_link_libraries(compare PUBLIC common)

# Smoke test: confirms the CUDA toolchain, GPU, and timing harness work.
add_executable(smoke apps/smoke.cu)
target_link_libraries(smoke PRIVATE common)

# Stages 1 and 2: color adjustment with the measurement harness.
add_executable(bench_color apps/bench_color.cu)
target_link_libraries(bench_color PRIVATE common image_io color_cpu color_cuda compare)

# Stage 3: correctness check.
add_executable(test_compare apps/test_compare.cpp)
target_link_libraries(test_compare PRIVATE compare)
add_executable(check_color apps/check_color.cu)
target_link_libraries(check_color PRIVATE common image_io color_cpu color_cuda compare)

# Stage 4: naive Gaussian blur.
add_library(blur_cpu STATIC src/blur_cpu.cpp)
target_link_libraries(blur_cpu PUBLIC common)
target_compile_options(blur_cpu PRIVATE -O2 -fno-tree-vectorize -fno-tree-slp-vectorize
                       -ffinite-math-only -fno-signed-zeros)
add_library(blur_cuda STATIC src/blur_cuda.cu)
target_link_libraries(blur_cuda PUBLIC common)
add_executable(bench_blur apps/bench_blur.cu)
target_link_libraries(bench_blur PRIVATE common image_io blur_cpu blur_cuda compare)
add_executable(check_blur apps/check_blur.cu)
target_link_libraries(check_blur PRIVATE common image_io blur_cpu blur_cuda compare)

enable_testing()
add_test(NAME compare_selftest COMMAND test_compare)
add_test(NAME color_cuda_vs_cpu COMMAND check_color WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(color_cuda_vs_cpu PROPERTIES ENVIRONMENT CHECK_IMPL=ctest)
add_test(NAME blur_cuda_vs_cpu COMMAND check_blur WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(blur_cuda_vs_cpu PROPERTIES ENVIRONMENT CHECK_IMPL=ctest)
