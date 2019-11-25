if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-Wall -Dcurrent_log_level=loglevel::DEBUG1)
endif ()

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(example_saxpy ${SOURCES} example_saxpy.cpp)
add_executable(example_program ${SOURCES} example_program.cpp)
add_executable(example_template ${SOURCES} example_template.cpp)