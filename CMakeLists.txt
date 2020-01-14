cmake_minimum_required(VERSION 3.09)
project(
  yacx
  VERSION 0.4.1
  LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

add_definitions(-DNVRTC_GET_TYPE_NAME=1)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-Wall -Dcurrent_log_level=loglevel::DEBUG)
endif()


include_directories(include)
include_directories(include/yacx)

link_libraries(cuda nvrtc)

file(GLOB SOURCES "src/*.cpp")
file(GLOB JNI_SOURCE_FILES "src/jni/*.cpp")
set(JAVA_SOURCE_FILES
    "src/java/BooleanArg.java"
    "src/java/Device.java"
    "src/java/DoubleArg.java"
    "src/java/Executor.java"
    "src/java/ExecutorFailureException.java"
    "src/java/FloatArg.java"
    "src/java/Headers.java"
    "src/java/IntArg.java"
    "src/java/JNIHandle.java"
    "src/java/Kernel.java"
    "src/java/KernelArg.java"
    "src/java/KernelTime.java"
    "src/java/LongArg.java"
    "src/java/Options.java"
    "src/java/Program.java"
    "src/java/ShortArg.java"
    "src/java/Utils.java")

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
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/java/bin/${_class_file}"
      COMMAND
        ${Java_JAVAC_EXECUTABLE} ${CMAKE_JAVA_COMPILE_FLAGS} -sourcepath
        "${PROJECT_SOURCE_DIR}/src/java" -d
        "${CMAKE_CURRENT_BINARY_DIR}/java/bin"
        "${PROJECT_SOURCE_DIR}/src/java/${_java_file}"
      DEPENDS "${PROJECT_SOURCE_DIR}/src/java/${_java_file}")
    list(APPEND class_files "java/bin/${_class_file}")
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
