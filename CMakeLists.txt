cmake_minimum_required(VERSION 3.9)
#cmake_minimum_required(VERSION 3.15)
project(cudacompiler VERSION 0.0.1)

enable_language(CXX CUDA)

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
file(GLOB TESTS "test/*.cpp")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test/test_header1.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test/test_header2.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test/test_pixel.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)

# EXAMPLES
add_executable(example_saxpy ${SOURCES} examples/example_saxpy.cpp)
add_executable(example_program ${SOURCES} examples/example_program.cpp)
add_executable(example_template ${SOURCES} examples/example_template.cpp)
target_compile_options(example_program PUBLIC -Wall -Dcurrent_log_level=loglevel::DEBUG1)
target_compile_options(example_template PUBLIC -Wall -Dcurrent_log_level=loglevel::DEBUG1)
target_compile_options(example_saxpy PUBLIC -Wall -Dcurrent_log_level=loglevel::DEBUG1)

## TESTS
include(ExternalProject)
ExternalProject_Add(
        catch
        PREFIX ${CMAKE_BINARY_DIR}/catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
)

# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(catch source_dir)
set(CATCH_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for Catch")

# Includes Catch in the project:
#add_subdirectory(${CMAKE_BINARY_DIR}/catch)
include_directories(${CATCH_INCLUDE_DIR})
enable_testing(true)

add_executable(tests ${SOURCES} ${TESTS})
#target_link_libraries(tests Catch2::Catch2)
