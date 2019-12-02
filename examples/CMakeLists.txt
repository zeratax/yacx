if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-Wall -Dcurrent_log_level=loglevel::INFO)
endif ()

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.cu
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/lena.ppm
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matrixMult.cu
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)


add_executable(example_saxpy ${SOURCES} example_saxpy.cpp)
add_executable(example_gauss ${SOURCES} example_gauss.cpp)
add_executable(example_program ${SOURCES} example_program.cpp)
add_executable(example_template ${SOURCES} example_template.cpp)
add_executable(example_matrix_multiply ${SOURCES} example_matrix_multiply.cpp)
