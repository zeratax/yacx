#include "../include/catch2/catch.hpp"

#include "../include/cudaexecutor/Options.hpp"
#include "../include/cudaexecutor/util.hpp"

#include <string>
#include <vector>

using cudaexecutor::to_comma_separated, cudaexecutor::type_of,
    cudaexecutor::load;

TEST_CASE("Vectors can be comma seperated",
          "[cudaexecutor::to_comma_seperated]") {
  std::vector<std::string> string_vec{"andre", "hasan", "jona", "felix"};
  std::vector<int> int_vec{1, 2, 3, 4, 5};
  std::vector<int> empty_vec;

  REQUIRE(to_comma_separated(string_vec.begin(), string_vec.end()) ==
          "andre, hasan, jona, felix");
  string_vec.pop_back();
  REQUIRE(to_comma_separated(string_vec.begin(), string_vec.end()) ==
          "andre, hasan, jona");
  REQUIRE(to_comma_separated(int_vec.begin(), int_vec.end()) ==
          "1, 2, 3, 4, 5");
  REQUIRE(to_comma_separated(empty_vec.begin(), empty_vec.end()) == "");
}

TEST_CASE("Displays the type of a variable", "[cudaexecutor::type_of]") {
  std::vector<int> vec;
  unsigned long long llui;
  cudaexecutor::Options options;

  REQUIRE(type_of(vec) == "std::vector<int, std::allocator<int> >");
  REQUIRE(type_of(options) == "cudaexecutor::Options");
  REQUIRE(type_of(llui) == "unsigned long long");
}

TEST_CASE("load file to std::string", "[cudaexecutor::load]") {
  std::string kernel{
      "extern \"C\" __global__\n"
      "void saxpy(float a, float *x, float *y, float *out, size_t n) {\n"
      "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (tid < n) {\n"
      "    out[tid] = a * x[tid] + y[tid];\n"
      "  }\n"
      "}\n"};

  REQUIRE(load("examples/kernels/saxpy.cu") == kernel);
}
