#include "../include/catch2/catch.hpp"

#include "../include/cudaexecutor/ProgramArg.hpp"

using cudaexecutor::ProgramArg;

TEST_CASE("ProgramArg can be constructed", "[cudaexecutor::ProgramArg]") {
  float a{5.1};
  std::array<float, 5> hX{1, 2, 3, 4, 5};
  std::array<float, 5> hY{6, 7, 8, 9, 10};
  std::array<float, 5> hOut{11, 12, 13, 14, 15};

  std::vector<ProgramArg> program_args;
  program_args.emplace_back(ProgramArg(&a));
  //    program_args.emplace_back(ProgramArg(hX.data(), bufferSize));
  //    program_args.emplace_back(ProgramArg(hY.data(), bufferSize));
  //    program_args.emplace_back(ProgramArg(hOut.data(), bufferSize));
  program_args.emplace_back(ProgramArg{&hX, bufferSize});
  program_args.emplace_back(ProgramArg{&hY, bufferSize});
  program_args.emplace_back(ProgramArg{&hOut, bufferSize, true});

  SECTION("ProgramArg can be uploaded and downloaded") {
    for (auto &arg : args)
      arg.upload();

    hX[0] = 0;
    hY[0] = 0;
    hOut[0] = 0;

    for (auto &arg : args)
      arg.download();
    REQUIRE(hX[0] == 0);
    REQUIRE(hY[0] == 0);
    REQUIRE(hOut[0] == 11);
  }
}
