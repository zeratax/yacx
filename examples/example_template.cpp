#include "../include/cudaexecutor/main.hpp"

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  int result{};
  try {
    Source source{"template<typename T>\n"
                    "__global__ void f3(int *result) { *result = sizeof(T); }"};

    std::vector<ProgramArg> program_args;
    program_args.push_back(ProgramArg{&result, sizeof(int), true});

    dim3 grid(1);
    dim3 block(1);
    Kernel test = source
            .program("f3")
                      .instantiate("int")
                      .compile()
                      .configure(grid, block)
                      .launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::cout << "Expected: " << sizeof(int) << " Actual: " << result << std::endl;
  return 0;
}
