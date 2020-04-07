file(GLOB TESTS "*.cpp")
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

file(COPY ${PROJECT_SOURCE_DIR}/examples/kernels/saxpy.cu
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/examples/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_header1.hpp
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_header2.hpp
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_pixel.hpp
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_compare.hpp
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/)

add_executable(tests ${SOURCES} ${TESTS})
set(catch2_dir ${PROJECT_SOURCE_DIR}/extern/catch2)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(
        FATAL_ERROR
          "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
      )
    endif()
  endif()
endif()

if(NOT EXISTS "${catch2_dir}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again."
  )
endif()

add_subdirectory(${catch2_dir} ${CMAKE_CURRENT_BINARY_DIR}/build)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${catch2_dir}/contrib/")

target_link_libraries(tests Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)

set(JAVA_TEST_FILES
	"test/java/yacx/TestArrayArg.java"
    "test/java/yacx/TestBooleanArg.java"
    "test/java/yacx/TestByteArg.java"
    "test/java/yacx/TestCProgram.java"
    "test/java/yacx/TestDevice.java"
    "test/java/yacx/TestDevices.java"
    "test/java/yacx/TestDoubleArg.java"
    "test/java/yacx/TestExecuteFilter.java"
    "test/java/yacx/TestExecuteSaxpy.java"
    "test/java/yacx/TestExecutor.java"
    "test/java/yacx/TestExecutorC.java"
    "test/java/yacx/TestFloatArg.java"
    "test/java/yacx/TestHeaders.java"
    "test/java/yacx/TestIntArg.java"
    "test/java/yacx/TestJNIHandle.java"
    "test/java/yacx/TestKernel.java"
    "test/java/yacx/TestLongArg.java"
    "test/java/yacx/TestOptions.java"
    "test/java/yacx/TestPaddingArg.java"
    "test/java/yacx/TestProgram.java"
    "test/java/yacx/TestShortArg.java"
    "test/java/yacx/TestUtils.java")

if(JNI_ENABLED)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/java/bin")
  set(class_files)
  include(ExternalProject)
  ExternalProject_Add(
          junit-platform-console
          PREFIX "${PROJECT_BINARY_DIR}/java/bin"
          URL https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.5.2/junit-platform-console-standalone-1.5.2.jar
          URL_HASH SHA512=3adfa07f95aad8f0104059b3b4256ea2e8f57b38fc6987551a8591a43849ac2953915d3f7eac71cac28a0b364d4a205dd80771860ab5d1757c14005e4f18400c
          DOWNLOAD_NO_EXTRACT 1
          DOWNLOAD_DIR ""
          CONFIGURE_COMMAND ""
          BUILD_COMMAND ""
          INSTALL_COMMAND ""
  )

  foreach(_java_path ${JAVA_TEST_FILES})
    get_filename_component(_java_file ${_java_path} NAME)
    string(REGEX REPLACE "\\.java$" ".class" _class_file "${_java_file}")
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/java/bin/${_class_file}"
      COMMAND
        ${Java_JAVAC_EXECUTABLE} ${CMAKE_JAVA_COMPILE_FLAGS}
            -sourcepath "${PROJECT_SOURCE_DIR}/src/main/java:${PROJECT_SOURCE_DIR}/test/java"
            -d "${PROJECT_BINARY_DIR}/java/bin"
            -cp "${PROJECT_BINARY_DIR}/java/bin/src/junit-platform-console-standalone-1.5.2.jar"
            "${PROJECT_SOURCE_DIR}/test/java/yacx/${_java_file}"
      DEPENDS "${PROJECT_SOURCE_DIR}/test/java/yacx/${_java_file}")
    list(APPEND class_files "${PROJECT_BINARY_DIR}/java/bin/${_class_file}")
  endforeach()
  add_custom_target(JNITestClasses ALL DEPENDS ${class_files} yacx-jni junit-platform-console)
endif()
