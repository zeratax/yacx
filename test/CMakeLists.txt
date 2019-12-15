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
SET(catch2_dir ${PROJECT_SOURCE_DIR}/extern/catch2)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${catch2_dir}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(${catch2_dir} ${CMAKE_CURRENT_BINARY_DIR}/build)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${catch2_dir}/contrib/")

target_link_libraries(tests Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)

SET(JAVA_TEST_FILES
        "test/java/TestBooleanArg.java"
        "test/java/TestDecice.java"
        "test/java/TestDoubleArg.java"
        "test/java/TestExecuteFilter.java"
        "test/java/TestExecuteSaxpy.java"
        "test/java/TestExecutor.java"
        "test/java/TestFloatArg.java"
        "test/java/TestHeaders.java"
        "test/java/TestIntArg.java"
        "test/java/TestJNIHandle.java"
        "test/java/TestKernel.java"
        "test/java/TestKernelArg.java"
        "test/java/TestKernelTime.java"
        "test/java/TestLongArg.java"
        "test/java/TestOptions.java"
        "test/java/TestProgram.java"
        "test/java/TestShortArg.java"
        "test/java/TestUtils.java"
        )

if (JNI_ENABLED)
    FILE(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/java/bin")
    SET(class_files)
    FOREACH (_java_path ${JAVA_TEST_FILES})
        get_filename_component(_java_file ${_java_path} NAME)
        STRING(REGEX REPLACE "\\.java$"
                ".class" _class_file
                "${_java_file}")
        ADD_CUSTOM_COMMAND(
                OUTPUT "${CMAKE_BINARY_DIR}/java/bin/${_class_file}"
                COMMAND
                ${Java_JAVAC_EXECUTABLE}
                ${CMAKE_JAVA_COMPILE_FLAGS}
                -sourcepath "${PROJECT_SOURCE_DIR}/src/java"
                -d "${CMAKE_BINARY_DIR}/java/bin"
                "${PROJECT_SOURCE_DIR}/test/java/${_java_file}"
                DEPENDS "${PROJECT_SOURCE_DIR}/test/java/${_java_file}")
        LIST(APPEND class_files "java/bin/${_class_file}")
    ENDFOREACH ()
    ADD_CUSTOM_TARGET(JavaJNITestClasses ALL DEPENDS ${class_files})
endif ()
