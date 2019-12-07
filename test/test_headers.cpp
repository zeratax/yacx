#include "yacx/Headers.hpp"

#include <catch2/catch.hpp>
#include <string.h>
#include <string>

using yacx::Headers, yacx::Header;

TEST_CASE("Header can be constructed", "[yacx::header]") {
  Header header("test_pixel.hpp");
  std::string content{"typedef struct {\n"
                      "  unsigned char r;\n"
                      "  unsigned char g;\n"
                      "  unsigned char b;\n"
                      "} Pixel;\n"};

  REQUIRE(std::string{header.name()} == std::string{"test_pixel.hpp"});
  REQUIRE(std::string{header.content()} == content);
}

TEST_CASE("Headers can be constructed", "[yacx::headers]") {
  Header header0("test_pixel.hpp");
  std::string content0{"typedef struct {\n"
                       "  unsigned char r;\n"
                       "  unsigned char g;\n"
                       "  unsigned char b;\n"
                       "} Pixel;\n"};
  Header header1("test_header1.hpp");
  std::string content1{"typedef struct {\n"
                       "  int x;\n"
                       "} header1;\n"};
  Header header2("test_header2.hpp");
  std::string content2{"typedef struct {\n"
                       "  int y;\n"
                       "} header2;\n"};

  SECTION("constructed from header") {
    Headers headers{header0, header1, header2};
    REQUIRE(headers.size() == 3);
    REQUIRE(std::string{headers.names()[2]} ==
            std::string{"test_pixel.hpp"});
    REQUIRE(std::string{headers.names()[1]} ==
            std::string{"test_header1.hpp"});
    REQUIRE(std::string{headers.names()[0]} ==
            std::string{"test_header2.hpp"});
    REQUIRE(std::string{headers.content()[2]} == content0);
    REQUIRE(std::string{headers.content()[1]} == content1);
    REQUIRE(std::string{headers.content()[0]} == content2);
  }
  SECTION("constructed from path") {
    Headers headers{"test_pixel.hpp", "test_header1.hpp",
                    "test_header2.hpp"};
    REQUIRE(headers.size() == 3);
    REQUIRE(std::string{headers.names()[2]} ==
            std::string{"test_pixel.hpp"});
    REQUIRE(std::string{headers.names()[1]} ==
            std::string{"test_header1.hpp"});
    REQUIRE(std::string{headers.names()[0]} ==
            std::string{"test_header2.hpp"});
    REQUIRE(std::string{headers.content()[2]} == std::string{content0});
    REQUIRE(std::string{headers.content()[1]} == std::string{content1});
    REQUIRE(std::string{headers.content()[0]} == std::string{content2});
  }
  SECTION("empty constructor, but inserted headers") {
    Headers headers{};
    REQUIRE(headers.size() == 0);
    headers.insert("test_pixel.hpp");
    REQUIRE(headers.size() == 1);
    headers.insert(header1);
    REQUIRE(headers.size() == 2);
    REQUIRE(std::string{headers.names()[0]} ==
            std::string{"test_pixel.hpp"});
    REQUIRE(std::string{headers.names()[1]} ==
            std::string{"test_header1.hpp"});
    REQUIRE(std::string{headers.content()[0]} == std::string{content0});
    REQUIRE(std::string{headers.content()[1]} == std::string{content1});
  }
}
