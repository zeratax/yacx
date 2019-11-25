file(GLOB TESTS "*.cpp")
file(GLOB SOURCES "../src/*.cpp")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_header1.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_header2.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_pixel.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/test)

include(FetchContent)
FetchContent_Declare(
        catch2
        PREFIX ${PROJECT_SOURCE_DIR}/lib/catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

add_executable(tests ${SOURCES} ${TESTS})
FetchContent_GetProperties(catch2)
if(NOT catch2_POPULATED)
    message(STATUS "downloading Catch2")
    FetchContent_Populate(catch2)
    add_subdirectory(${catch2_SOURCE_DIR} ${catch2_BINARY_DIR})
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${catch2_SOURCE_DIR}/contrib/")
enable_testing(true)
target_link_libraries(tests Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)