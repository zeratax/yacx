#include "../include/catch2/catch.hpp"

#include "../include/cudaexecutor/Headers.hpp"

#include <string.h>
#include <string>

using cudaexecutor::Headers, cudaexecutor::Header;

TEST_CASE("Header can be constructed", "[cudaexecutor::header]") {
  Header header("test/test_pixel.hpp");
  const char *content = "typedef struct {\n"
                        "  unsigned char r;\n"
                        "  unsigned char g;\n"
                        "  unsigned char b;\n"
                        "} Pixel;\n";

  REQUIRE(strcmp(header.name(), "test/test_pixel.hpp") == 0);
  REQUIRE(strcmp(header.content(), content) == 0);
}

TEST_CASE("Headers can be constructed", "[cudaexecutor::headers]") {
  Header header0("test/test_pixel.hpp");
  const char *content0 = "typedef struct {\n"
                         "  unsigned char r;\n"
                         "  unsigned char g;\n"
                         "  unsigned char b;\n"
                         "} Pixel;\n";
  Header header1("test/test_header1.hpp");
  const char *content1 = "typedef struct {\n"
                         "  int x;\n"
                         "} header1;\n";
  Header header2("test/test_header2.hpp");
  const char *content2 = "typedef struct {\n"
                         "  int y;\n"
                         "} header2;\n";

  SECTION("constructed from header") {
    Headers headers{header0, header1, header2};
    REQUIRE(headers.size() == 3);
    REQUIRE(strcmp(headers.names()[2], "test/test_pixel.hpp") == 0);
    REQUIRE(strcmp(headers.names()[1], "test/test_header1.hpp") == 0);
    REQUIRE(strcmp(headers.names()[0], "test/test_header2.hpp") == 0);
    REQUIRE(strcmp(headers.content()[2], content0) == 0);
    REQUIRE(strcmp(headers.content()[1], content1) == 0);
    REQUIRE(strcmp(headers.content()[0], content2) == 0);
  }
  SECTION("constructed from path") {
    Headers headers{"test/test_pixel.hpp", "test/test_header1.hpp",
                    "test/test_header2.hpp"};
    REQUIRE(headers.size() == 3);
    REQUIRE(strcmp(headers.names()[2], "test/test_pixel.hpp") == 0);
    REQUIRE(strcmp(headers.names()[1], "test/test_header1.hpp") == 0);
    REQUIRE(strcmp(headers.names()[0], "test/test_header2.hpp") == 0);
    REQUIRE(strcmp(headers.content()[2], content0) == 0);
    REQUIRE(strcmp(headers.content()[1], content1) == 0);
    REQUIRE(strcmp(headers.content()[0], content2) == 0);
  }
  SECTION("empty constructor, but inserted headers") {
    Headers headers{};
    REQUIRE(headers.size() == 0);
    headers.insert("test/test_pixel.hpp");
    REQUIRE(headers.size() == 1);
    headers.insert(header1);
    REQUIRE(headers.size() == 2);
    REQUIRE(strcmp(headers.names()[0], "test/test_pixel.hpp") == 0);
    REQUIRE(strcmp(headers.names()[1], "test/test_header1.hpp") == 0);
    REQUIRE(strcmp(headers.content()[0], content0) == 0);
    REQUIRE(strcmp(headers.content()[1], content1) == 0);
  }
}
