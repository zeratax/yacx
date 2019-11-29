#include "../include/cudaexecutor/main.hpp"
#include <experimental/iterator>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of;

template<int N, typename T>
void my_kernel(T* data) {
  T data0 = data[0];
  for( int i{0}; i<N; ++i ) {
    data[i] *= data0;
  }
}

int main() {
  std::array<int, 10> array;
  array.fill(0);
  int data{8};
  try {
    Device device;
    Options options{cudaexecutor::options::GpuArchitecture(device),
                    cudaexecutor::options::FMAD(false)};
    Source source{"template<typename type, int size>\n"
                  "__global__ void my_kernel(type[] c, type val) {\n"
                  "    auto idx = threadIdx.x * size;\n"
                  "\n"
                  "    #pragma unroll(size)\n"
                  "    for (auto i = 0; i < size; i++) {\n"
                  "        c[idx] = val;\n"
                  "        idx++;\n"
                  "    }\n"
                  "}"};

    std::vector<ProgramArg> program_args;
    program_args.emplace_back(ProgramArg{array.data(), sizeof(int) * 5, true});
    program_args.emplace_back(ProgramArg{&data});

    dim3 grid(1);
    dim3 block(1);
    source
        .program("my_kernel")
        .instantiate(5, type_of(data))
        .compile(options)
        .configure(grid, block)
        .launch(program_args, device);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::copy(array.begin(), array.end(),
            std::experimental::make_ostream_joiner(std::cout, ", "));
  std::cout << std::endl;

  return 0;
}
