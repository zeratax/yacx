#include "cudaexecutor/Headers.hpp"
#include "cudaexecutor/Kernel.hpp"
#include "cudaexecutor/KernelArg.hpp"
#include "cudaexecutor/Program.hpp"
#include "cudaexecutor/Source.hpp"
#include "test_compare.hpp"

#include <catch2/catch.hpp>
#include <iostream>

using cudaexecutor::KernelArg, cudaexecutor::Source, cudaexecutor::Headers;

CATCH_REGISTER_ENUM(compare, compare::CORRECT, compare::CHECK_COMPARE_WRONG,
                    compare::A_COMPARE_WRONG, compare::X_COMPARE_WRONG,
                    compare::Y_COMPARE_WRONG, compare::OUT_COMPARE_WRONG);

TEST_CASE("KernelArg can be constructed", "[cudaexecutor::KernelArg]") {
  int a{5};
  compare check{CORRECT};
  int *hX = new int[5]{1, 2, 3, 4, 5};
  int *hY = new int[5]{6, 7, 8, 9, 10};
  int *hOut = new int[5]{11, 12, 13, 14, 15};
  size_t bufferSize = 5 * sizeof(int);

  std::vector<KernelArg> args;
  args.emplace_back(KernelArg(&a));
  args.emplace_back(KernelArg{&hX, bufferSize});
  args.emplace_back(KernelArg{&hY, bufferSize});
  args.emplace_back(KernelArg{&hOut, bufferSize, true});
  args.emplace_back(KernelArg{&check, sizeof(compare), true});

  SECTION("KernelArg can be downloaded") {
    REQUIRE(a == 5);
    REQUIRE(hX[0] == 1);
    REQUIRE(hY[0] == 6);
    REQUIRE(hOut[0] == 11);
    Source source{"#include \"test/test_compare.hpp\"\n"
                  "extern \"C\"\n"
                  "__global__ void download(int a, int *x, int *y, int "
                  "*out, compare *check) {\n"
                  "  a = 6;\n"
                  "  x[0] = 2;\n"
                  "  y[0] = 7;\n"
                  //"  out[0] = 12;\n" // => SIGSEGV
                  "}",
                  Headers{"test/test_compare.hpp"}};

    dim3 grid(1);
    dim3 block(1);
    source.program("download").compile().configure(grid, block).launch(args);

    REQUIRE(a == 5);
    REQUIRE(hX[0] == 1);
    REQUIRE(hY[0] == 6);
    REQUIRE(hOut[0] == 12);
  }
  SECTION("KernelArg can be uploaded") {
    REQUIRE(hX[0] == 1);
    REQUIRE(hY[0] == 6);
    REQUIRE(hOut[0] == 11);
    Source source{"#include \"test/test_compare.hpp\"\n"
                  "extern \"C\"\n"
                  "__global__ void compare(int a, int *x, int *y, int "
                  "*out, compare *check) {\n"
                  "  int dA = 5.1;\n"
                  "  int dX[5] = {1, 2, 3, 4, 5};\n"
                  "  int dY[5] = {6, 7, 8, 9, 10};\n"
                  "  int dOut[5] = {11, 12, 13, 14, 15};\n"
                  "  if(*check != CORRECT) {\n"
                  "    *check = CHECK_COMPARE_WRONG;\n"
                  "  } else if (dA != a) {\n"
                  "    *check = A_COMPARE_WRONG;\n"
                  "  } else {\n"
                  "    for (int i = 0; i < 5; ++i) {\n"
                  "      if(dX[i] != x[i]) {\n"
                  "        *check = X_COMPARE_WRONG;\n"
                  "        break;\n"
                  "      } else if(dY[i] == y[i]) {\n"
                  "        *check = Y_COMPARE_WRONG;\n"
                  "        break;\n"
                  "      } else if(dOut[i] != out[i]) {\n"
                  "        *check = OUT_COMPARE_WRONG;\n"
                  "        break;\n"
                  "      }\n"
                  "    }\n"
                  "  }\n"
                  "}",
                  Headers{"test/test_compare.hpp"}};

    dim3 grid(1);
    dim3 block(1);
    source.program("compare").compile().configure(grid, block).launch(args);
    REQUIRE(check == CORRECT);
  }

  delete[] hX;
  delete[] hY;
  delete[] hOut;
}
