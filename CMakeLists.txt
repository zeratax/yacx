cmake_minimum_required(VERSION 3.11)
project(cudaexecutor VERSION 0.0.1)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_definitions(-DNVRTC_GET_TYPE_NAME=1)

include_directories(include)

link_directories("$ENV{CUDA_PATH}/lib64")
link_libraries(cuda nvrtc)

set(SOURCES ${PROJECT_SOURCE_DIR}/src/Device.cpp, ${PROJECT_SOURCE_DIR}/src/Headers.cpp, ${PROJECT_SOURCE_DIR}/src/Exception.cpp, ${PROJECT_SOURCE_DIR}/src/Kernel.cpp, ${PROJECT_SOURCE_DIR}/src/Options.cpp, ${PROJECT_SOURCE_DIR}/src/Program.cpp, ${PROJECT_SOURCE_DIR}/src/ProgramArg.cpp, ${PROJECT_SOURCE_DIR}/src/Source.cpp, ${PROJECT_SOURCE_DIR}/src/util.cpp)

add_subdirectory(test)