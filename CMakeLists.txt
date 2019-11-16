cmake_minimum_required(VERSION 3.9)
#cmake_minimum_required(VERSION 3.15)
project(cudacompiler VERSION 0.0.1)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

#add_compile_definitions(NVRTC_GET_TYPE_NAME=1) # requires cmake 3.12
add_definitions(-DNVRTC_GET_TYPE_NAME=1)

enable_testing()

include_directories(include)
include_directories(include/cudaexecutor)
include_directories("$ENV{CUDA_PATH}/include")

link_directories("$ENV{CUDA_PATH}/lib64")
link_libraries(cuda nvrtc)

file(GLOB SOURCES "src/*.cpp")

# EXAMPLES
add_executable(example_program ${SOURCES} examples/example_program.cpp)
target_compile_options(example_program PUBLIC -Wall)

## TESTS
#add_subdirectory(lib/Catch2)
#find_package(Catch2 REQUIRED)

set(CATCH_INCLUDE_DIR include/catch2)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

file(GLOB TEST_SOURCES "test/*.cpp")
add_executable(tests ${SOURCES} ${TEST_SOURCES})
target_link_libraries(tests Catch)