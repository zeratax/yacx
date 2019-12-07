#include "yacx/Options.hpp"

#include <catch2/catch.hpp>
#include <string.h>

using yacx::Options;

TEST_CASE("Options can be constructed", "[yacx::Options]") {
  Options options1({yacx::options::GpuArchitecture(30, 5),
                    yacx::options::Fast_Math(false)});
  Options options2;
  const char *const options_string[] = {"--gpu-architecture=compute_305",
                                        "--use_fast_math=false"};

  REQUIRE(options1.numOptions() == 2);
  REQUIRE(strcmp(options1.options()[1], options_string[1]) == 0);
  REQUIRE(strcmp(options1.options()[0], options_string[0]) == 0);
  REQUIRE(options2.numOptions() == 0);
}

TEST_CASE("Options can be inserted", "[yacx::Options]") {
  Options options;
  const char *options_string[] = {"--gpu-architecture=compute_400",
                                  "--fmad=true"};

  REQUIRE(options.numOptions() == 0);
  options.insertOptions(yacx::options::GpuArchitecture(40, 0),
                        yacx::options::FMAD(true));
  REQUIRE(options.numOptions() == 2);
  REQUIRE(strcmp(options.options()[0], options_string[0]) == 0);
  REQUIRE(strcmp(options.options()[1], options_string[1]) == 0);
}
