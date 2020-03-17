cmake_minimum_required(VERSION 3.09)
project(
  yacx
  VERSION 0.5.0
  LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_definitions(-Wall -Wextra --pedantic)
add_definitions(-DNVRTC_GET_TYPE_NAME=1)

include_directories(include)
include_directories(include/yacx)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

link_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib")

link_libraries(cuda nvrtc)


file(GLOB SOURCES "src/*.cpp" "src/cexecutor/*.cpp")
file(GLOB JNI_SOURCE_FILES "src/jni/*.cpp")
set(JAVA_SOURCE_FILES
    "src/main/java/yacx/ArrayArg.java"
    "src/main/java/yacx/BooleanArg.java"
    "src/main/java/yacx/ByteArg.java"
    "src/main/java/yacx/CProgram.java"
    "src/main/java/yacx/Device.java"
    "src/main/java/yacx/Devices.java"
    "src/main/java/yacx/DoubleArg.java"
    "src/main/java/yacx/Executor.java"
    "src/main/java/yacx/ExecutorFailureException.java"
    "src/main/java/yacx/FloatArg.java"
    "src/main/java/yacx/HalfArg.java"
    "src/main/java/yacx/Headers.java"
    "src/main/java/yacx/IntArg.java"
    "src/main/java/yacx/JNIHandle.java"
    "src/main/java/yacx/Kernel.java"
    "src/main/java/yacx/KernelArg.java"
    "src/main/java/yacx/KernelTime.java"
    "src/main/java/yacx/LongArg.java"
    "src/main/java/yacx/Options.java"
    "src/main/java/yacx/PaddingArg.java"
    "src/main/java/yacx/Program.java"
    "src/main/java/yacx/ShortArg.java"
    "src/main/java/yacx/Utils.java")

add_library(yacx SHARED ${SOURCES})

option(JNI_ENABLED "Build JNI" ON)
if(JNI_ENABLED)
  message(STATUS "JNI enabled")
  find_package(Java REQUIRED)
  include(UseJava)

  # Build Java classes
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/java/bin")
  set(class_files)
  foreach(_java_path ${JAVA_SOURCE_FILES})
    get_filename_component(_java_file ${_java_path} NAME)
    # _java_file: relative file name _class_file: relative class name
    string(REGEX REPLACE "\\.java$" ".class" _class_file "${_java_file}")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/java/bin/yacx/${_class_file}"
      COMMAND
        ${Java_JAVAC_EXECUTABLE} ${CMAKE_JAVA_COMPILE_FLAGS} -sourcepath
        "${PROJECT_SOURCE_DIR}/src/main/java/" -d
        "${CMAKE_CURRENT_BINARY_DIR}/java/bin"
        "${PROJECT_SOURCE_DIR}/src/main/java/yacx/${_java_file}"
      DEPENDS "${PROJECT_SOURCE_DIR}/src/main/java/yacx/${_java_file}")
    list(APPEND class_files "java/bin/yacx/${_class_file}")
  endforeach()
  add_custom_target(JNIClasses ALL DEPENDS ${class_files})

  find_package(JNI REQUIRED)

  include_directories(${JNI_INCLUDE_DIRS})
  add_library(yacx-jni SHARED ${JNI_SOURCE_FILES})
  add_dependencies(yacx-jni yacx JNIClasses)
  target_link_libraries(yacx-jni yacx ${JNI_LIBRARIES})
  set_target_properties(
    yacx-jni
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
               LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
               RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
  set(PACKAGE_TEST_DEFAULT ON)
else()
  set(PACKAGE_TEST_DEFAULT OFF)
endif()

option(PACKAGE_TESTS "Build the tests" ${PACKAGE_TEST_DEFAULT})
if(PACKAGE_TESTS)
  message(STATUS "Testing enabled")
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(examples)
else()
  message(STATUS "testing not enabled")
endif()
