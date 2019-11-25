cmake_minimum_required(VERSION 3.09)
project(cudaexecutor VERSION 0.0.1)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_definitions(-DNVRTC_GET_TYPE_NAME=1)

include_directories(include)

include_directories("$ENV{CUDA_PATH}/include")
link_directories("$ENV{CUDA_PATH}/lib64") # there are windows specific paths for this
link_libraries(cuda nvrtc)

#file(GLOB SOURCES "src/*.cpp")
set(SOURCES ${PROJECT_SOURCE_DIR}/src/Device.cpp, ${PROJECT_SOURCE_DIR}/src/Headers.cpp, ${PROJECT_SOURCE_DIR}/src/Exception.cpp, ${PROJECT_SOURCE_DIR}/src/Kernel.cpp, ${PROJECT_SOURCE_DIR}/src/Options.cpp, ${PROJECT_SOURCE_DIR}/src/Program.cpp, ${PROJECT_SOURCE_DIR}/src/ProgramArg.cpp, ${PROJECT_SOURCE_DIR}/src/Source.cpp, ${PROJECT_SOURCE_DIR}/src/util.cpp)

set(PACKAGE_TEST_DEFAULT ${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
option(PACKAGE_TESTS "Build the tests" PACKAGE_TEST_DEFAULT)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
add_subdirectory(examples)
