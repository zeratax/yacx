cmake_minimum_required(VERSION 3.9)
#cmake_minimum_required(VERSION 3.15)
project(cudacompiler VERSION 0.0.1)

enable_language(CXX CUDA)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

#add_compile_definitions(NVRTC_GET_TYPE_NAME=1) # requires cmake 3.12
add_definitions(-DNVRTC_GET_TYPE_NAME=1)

include_directories(include)
include_directories(include/cudaexecutor)
include_directories("$ENV{CUDA_PATH}/include")

link_directories("$ENV{CUDA_PATH}/lib64")
link_libraries(cuda nvrtc)

file(GLOB SOURCES "src/*.cpp")

# EXAMPLES
add_executable(example_saxpy ${SOURCES} examples/example_saxpy.cpp)
add_executable(example_program ${SOURCES} examples/example_program.cpp)
target_compile_options(example_program PUBLIC -Wall)
target_compile_options(example_saxpy PUBLIC -Wall -Dcurrent_log_level=loglevel::DEBUG)

## TESTS
#add_subdirectory(lib/Catch2)
#find_package(Catch2 REQUIRED)

ExternalProject_Add(
        catch
        PREFIX ${CMAKE_BINARY_DIR}/catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        TIMEOUT 10
        UPDATE_COMMAND ${GIT_EXECUTABLE} pull
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
)

# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(catch source_dir)
set(CATCH_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for Catch")
#set(CATCH_INCLUDE_DIR include/catch2)

enable_testing()

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

file(GLOB TEST_SOURCES "test/*.cpp")
add_executable(tests ${SOURCES} ${TEST_SOURCES})
target_link_libraries(tests Catch)