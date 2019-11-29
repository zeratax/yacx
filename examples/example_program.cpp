#include "cudaexecutor/main.hpp"
#include <experimental/iterator>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of;

int main() {
  std::array<int, 32> array;
  array.fill(0);
  int data{1};
  try {
    Device device;
    Options options{cudaexecutor::options::GpuArchitecture(device),
                    cudaexecutor::options::FMAD(false)};
    options.insert("--std", "c++14");
    Source source{
        "template<typename type, int size>\n"
        "__global__ void my_kernel(type* c, type val) {\n"
        "    auto idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "\n"
        "    #pragma unroll(size)\n"
        "    for (auto i = idx * size; i < idx * size + size; i++) {\n"
        "        c[i] = idx + val;\n"
        "    }\n"
        "}"};

    std::vector<ProgramArg> program_args;
    program_args.emplace_back(
        ProgramArg{array.data(), sizeof(int) * array.size(), true});
    program_args.emplace_back(ProgramArg{&data});

    dim3 grid(8);
    dim3 block(1);
    source.program("my_kernel")
        .instantiate(type_of(data), 4)
        .compile(options)
        .configure(grid, block)
        .launch(program_args, device);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::cout << '\n';
  std::copy(array.begin(), array.end(),
            std::experimental::make_ostream_joiner(std::cout, ", "));
  std::cout << std::endl;

  return 0;
}
