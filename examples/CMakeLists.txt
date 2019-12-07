if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-Wall -Dcurrent_log_level=loglevel::INFO)
endif ()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gauss.cu
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/lena.ppm
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/matrixMult.cu
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/kernels)

link_libraries(cudaexecutor)

add_executable(example_saxpy example_saxpy.cpp)
add_executable(example_gauss example_gauss.cpp)
add_executable(example_program example_program.cpp)
add_executable(example_template example_template.cpp)
add_executable(example_matrix_multiply example_matrix_multiply.cpp)

SET(JAVA_EXAMPLE_FILES
        "java/ExampleSaxpy.java"
        "java/ExampleFilterExecutor.java"
        "java/ExampleBlockReduce.java")

if (JNI_ENABLED)
    FOREACH (_java_path ${JAVA_EXAMPLE_FILES})
        get_filename_component(_java_file ${_java_path} NAME)
        get_filename_component(_java_name ${_java_path} NAME_WE)
        # _java_file: relative file name
        # _class_file: relative class name
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
            "${PROJECT_SOURCE_DIR}/examples/java/${_java_file}"
            DEPENDS "${PROJECT_SOURCE_DIR}/examples/java/${_java_file}")
        ADD_CUSTOM_TARGET(${_java_name} ALL DEPENDS cudaexecutor-jni
            ${CMAKE_BINARY_DIR}/java/bin/${_class_file})
    ENDFOREACH ()
    #    ADD_TEST(
    #            NAME run-jni-example-saxpy
    #            COMMAND
    #            "${Java_JAVA_EXECUTABLE}"
    #            -ea
    #            -Djava.library.path=${CMAKE_BINARY_DIR}
    #            ExampleSaxpy
    #            WORKING_DIRECTORY "${CMAKE_BINARY_DIR} ")
endif ()
