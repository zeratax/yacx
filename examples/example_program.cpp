#include "yacx/main.hpp"
#include <experimental/iterator>

using yacx::Source, yacx::KernelArg, yacx::Kernel,
    yacx::Options, yacx::Device, yacx::load,
    yacx::type_of;

int main() {
  std::array<int, 32> array;
  array.fill(0);
  int data{1};
  try {
    Device device;
    Options options{yacx::options::GpuArchitecture(device),
                    yacx::options::FMAD(false)};
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

    std::vector<KernelArg> args;
    args.emplace_back(
        KernelArg{array.data(), sizeof(int) * array.size(), true});
    args.emplace_back(KernelArg{&data});

    dim3 grid(8);
    dim3 block(1);
    source.program("my_kernel")
        .instantiate(type_of(data), 4)
        .compile(options)
        .configure(grid, block)
        .launch(args, device);
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
