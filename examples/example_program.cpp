#include "yacx/main.hpp"
#include <algorithm>
#include <experimental/iterator>
#include <vector>

using yacx::Source, yacx::KernelArg, yacx::Options, yacx::Device, yacx::type_of;

int main(int argc, char const *const *const argv) {
  yacx::handle_logging_args(argc, argv);
  const int data{1};
  const int times{4};
  const size_t size{32};

  static_assert(!(size % times));

  std::vector<int> v;
  v.resize(size);
  std::fill(v.begin(), v.end(), 0);

  try {
    Device device = Devices::findDevice();
    Options options{yacx::options::GpuArchitecture(device),
                    yacx::options::FMAD(false)};
    options.insert("--std", "c++14");
    Source source{"template<typename type, int size>\n"
                  "__global__ void my_kernel(type* c, type val) {\n"
                  "    auto idx{blockIdx.x * blockDim.x + threadIdx.x};\n"
                  "\n"
                  "    #pragma unroll(size)\n"
                  "    for (auto i{0}; i < size; ++i) {\n"
                  "        c[i] = idx + val;\n"
                  "    }\n"
                  "}"};

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{v.data(), sizeof(int) * v.size(), true});
    args.emplace_back(KernelArg{const_cast<int *>(&data)});

    dim3 grid(v.size() / times);
    dim3 block(1);
    source.program("my_kernel")
        .instantiate(type_of(data), times)
        .compile(options)
        .configure(grid, block)
        .launch(args, device);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::cout << '\n';
  std::copy(v.begin(), v.end(),
            std::experimental::make_ostream_joiner(std::cout, ", "));
  std::cout << std::endl;

  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7,
  // 7, 7, 7, 8, 8, 8, 8

  return 0;
}
