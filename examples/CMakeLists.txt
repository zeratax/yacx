if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-Wall -Wextra --pedantic -Dcurrent_log_level=loglevel::INFO)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.h
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.cu
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/lena.ppm
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matrixMult.cu
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)

link_libraries(yacx)

add_executable(example_saxpy example_saxpy.cpp)
add_executable(example_gauss example_gauss.cpp)
add_executable(example_program example_program.cpp)
add_executable(example_template example_template.cpp)
add_executable(example_matrix_multiply example_matrix_multiply.cpp)
add_executable(example_logger example_logger.cpp)

set(JAVA_EXAMPLE_FILES
    "java/ExampleCGEMM.java"
    "java/ExampleDotProduct.java"
    "java/ExampleFastGEMM.java"
    "java/ExampleFastGEMMBenchmark"
    "java/ExampleFilter.java"
    "java/ExampleFilterBenchmark.java"
    "java/ExampleFilterExecutor.java"
    "java/ExampleGauss.java"
    "java/ExampleReduce.java"
    "java/ExampleReduceBenchmark.java"
    "java/ExampleSaxpy.java"
    "java/ExampleSaxpyBenchmark.java"
    "java/ExampleSaxpyExecutor.java"
    "java/ExampleSimpleGEMM.java"
    "java/ExampleSimpleGEMMBenchmark.java"
    "java/TemplateKernel.java")

if(JNI_ENABLED)
  foreach(_java_path ${JAVA_EXAMPLE_FILES})
    get_filename_component(_java_file ${_java_path} NAME)
    get_filename_component(_java_name ${_java_path} NAME_WE)
    # _java_file: relative file name _class_file: relative class name
    string(REGEX REPLACE "\\.java$" ".class" _class_file "${_java_file}")
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/java/bin/${_class_file}"
      COMMAND
        ${Java_JAVAC_EXECUTABLE} ${CMAKE_JAVA_COMPILE_FLAGS} -sourcepath
        "${PROJECT_SOURCE_DIR}/src/java" -d "${CMAKE_BINARY_DIR}/java/bin"
        "${PROJECT_SOURCE_DIR}/examples/java/${_java_file}"
      DEPENDS "${PROJECT_SOURCE_DIR}/examples/java/${_java_file}")
    add_custom_target(
      ${_java_name} ALL DEPENDS yacx-jni
                                ${CMAKE_BINARY_DIR}/java/bin/${_class_file})
  endforeach()
  # ADD_TEST( NAME run-jni-example-saxpy COMMAND "${Java_JAVA_EXECUTABLE}" -ea
  # -Djava.library.path=${CMAKE_BINARY_DIR} ExampleSaxpy WORKING_DIRECTORY
  # "${CMAKE_BINARY_DIR} ")
endif()
