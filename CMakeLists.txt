cmake_minimum_required(VERSION 3.09)
project(cudaexecutor VERSION 0.2.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_definitions(-DNVRTC_GET_TYPE_NAME=1)

include_directories(include)
include_directories(include/cudaexecutor)

link_libraries(cuda nvrtc)

file(GLOB SOURCES "src/*.cpp")
file(GLOB JNI_SOURCE_FILES "src/jni/*.cpp")
SET(JAVA_SOURCE_FILES
        "src/java/ArrayArg.java"
        "src/java/Device.java"
        "src/java/Executor.java"
        "src/java/ExecutorFailureException.java"
        "src/java/Headers.java"
        "src/java/JNIHandle.java"
        "src/java/Kernel.java"
        "src/java/KernelArg.java"
        "src/java/Options.java"
        "src/java/Program.java"
        "src/java/Utils.java"
        "src/java/ValueArg.java")

add_library(cudaexecutor SHARED ${SOURCES})

option(JNI_ENABLED "Build JNI" ON)
if (JNI_ENABLED)
    message(STATUS "JNI enabled")
    find_package(Java REQUIRED)
    include(UseJava)

    # Build Java classes
    FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/java/bin")
    SET(class_files)
    FOREACH (_java_path ${JAVA_SOURCE_FILES})
        get_filename_component(_java_file ${_java_path} NAME)
        # _java_file: relative file name
        # _class_file: relative class name
        STRING(REGEX REPLACE "\\.java$"
                ".class" _class_file
                "${_java_file}")
        ADD_CUSTOM_COMMAND(
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/java/bin/${_class_file}"
                COMMAND
                ${Java_JAVAC_EXECUTABLE}
                ${CMAKE_JAVA_COMPILE_FLAGS}
                -sourcepath "${PROJECT_SOURCE_DIR}/src/java"
                -d "${CMAKE_CURRENT_BINARY_DIR}/java/bin"
                "${PROJECT_SOURCE_DIR}/src/java/${_java_file}"
                DEPENDS "${PROJECT_SOURCE_DIR}/src/java/${_java_file}")
        LIST(APPEND class_files "java/bin/${_class_file}")
    ENDFOREACH ()
    ADD_CUSTOM_TARGET(JavaJNIClasses ALL DEPENDS ${class_files})

    FIND_PACKAGE(JNI REQUIRED)

    include_directories(${JNI_INCLUDE_DIRS})
    add_library(cudaexecutor-jni SHARED ${JNI_SOURCE_FILES})
    target_link_libraries(cudaexecutor-jni cudaexecutor ${JNI_LIBRARIES})
    SET_TARGET_PROPERTIES(
            cudaexecutor-jni PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
endif ()


if (${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
    set(PACKAGE_TEST_DEFAULT ON)
else ()
    set(PACKAGE_TEST_DEFAULT OFF)
endif ()

option(PACKAGE_TESTS "Build the tests" ${PACKAGE_TEST_DEFAULT})
if (PACKAGE_TESTS)
    message(STATUS "Testing enabled")
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(examples)
else ()
    message(STATUS "testing not enabled")
endif ()
