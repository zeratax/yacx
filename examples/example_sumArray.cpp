#include "yacx/main.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

using yacx::Headers, yacx::Header, yacx::Source, yacx::KernelArg, yacx::Kernel, yacx::Options, 
yacx::Device,yacx::type_of, yacx::KernelTime, yacx::Devices, yacx::arg_type, yacx::load;

//A function template to compare the values of two array elements.
template<typename type>
void compare(type *lhs, type *rhs, int array_size) {
  //1. Declaring the variables prior to comparing 2 array elements
  int errors{0};
  float lhs_type, rhs_type;
  std::string errors_string, array_size_string, i_string, lhs_string, 
  rhs_string;

  /*2A. Calculate the number of errors caused by inconsistencies between
        values of two array elements.*/
  for (int i{0}; i < array_size; i += 1) {
    lhs_type=static_cast<int>(lhs[i]);
    rhs_type=static_cast<int>(rhs[i]);

    if ((lhs_type-rhs_type)!= 0) {
      errors += 1;
      i_string=std::to_string(i);
      lhs_string=std::to_string(lhs[i]);
      rhs_string=std::to_string(rhs[i]);

      std::cout << i_string << " expected " << lhs_string << ": actually "
      << rhs_string << std::endl;
    }
  }

  /*2B. Determining, whether there are errors in the consistencies of values
        of two array elements.*/
  errors_string=static_cast<int>(errors);
  array_size_string=static_cast<int>(array_size);

  if (errors > 0){
    std::cout << "\u001b[31m" << errors_string << " errors occured, out of "
    << array_size_string << " values.\u001b[0m" << std::endl;
  }else{
    std::cout << "\u001b[32mNo errors occured.\u001b[0m" << std::endl;
  }
}

//Template function for comparing the values of two array elements
template<typename type>
std::function<bool(const type &, const type &)> comparator =
    [](const type &left, const type &right) {
      double epsilon{1.0E-8};
      float lhs_type=static_cast<float>(left), rhs_type=static_cast<float>(right);
      return (abs(lhs_type- rhs_type) < epsilon);
};

//A template function to fill all array elements with random values
template<typename type>
void fillArray(type *array, int array_size, int min=0,int max=100){
  //1. Declaring and initialising variables required to run the template function
  static std::random_device rd;
  static std::mt19937 mte(rd());
  std::uniform_int_distribution<int> dist(min, max);

  //2. Generating random values using random value generator
  for(int i=0;i<array_size;++i)
    array[i] = static_cast<type>(dist(mte));
}

//Template function to calculate 2 arrays of similar length on the host-side.
template<typename type>
void sumArrayOnHost(const type *A,const type *B,type *C,
const size_t nx,const size_t ny,const size_t nz ){
  //1. Declaring the variables required to add values from two 3D-array elements
  size_t i;

  //2. Loop to calculate the sum of values of two array elements
  for(size_t ix=0;ix<nx;++ix){
    for(size_t iy=0;iy<ny;++iy){
      for(size_t iz=0;iz<nz;++iz){
        i = iz*(nz*ny)+iy*ny+ix;
        C[i]=A[i]+B[i];
      }
    }
  }
}

//A general example of the functionalities of a CUDA-Executor
int main(){
  /*1. Declaring and initialising the variables required to run the 
       CUDA-Executor-example*/
  std::clock_t start;
  bool equalSumArray;
  const float data{1.69f};
  const size_t NX{16},NY{8},NZ{8};
  const size_t BLOCK_SIZE{8};
  const size_t matrix_size{(NX*NY*NZ) * sizeof(float)};

  /*1A. Checking the correctness of the modulus of NX,NY,NZ with
        BLOCK_SIZE*/
  static_assert(NX % BLOCK_SIZE == 0);
  static_assert(NY % BLOCK_SIZE == 0);
  static_assert(NZ % BLOCK_SIZE == 0);

  float *A = new float[NX*NY*NZ];
  float *B = new float[NX*NY*NZ];
  float *C_seq = new float[NX*NY*NZ];
  float *C_cuda = new float[NX*NY*NZ];

  KernelTime time;

  //1B. Fill arrays with random values
  fillArray<float>(A,NX*NY*NZ);
  fillArray<float>(B,NX*NY*NZ);

  //2. Running the CUDA-Executor-example
  try {
    //2A. Select Device
    Device dev = Devices::findDevice();
    std::cout << "===================================\n";
    std::cout << "Selected " << dev.name() << " with "
              << (dev.total_memory() / 1024) / 1024 << "mb VRAM\n";
    std::cout << "Kernel Arguments total size: "
              << ((matrix_size * 3 + sizeof(size_t)) / 1024) << "kb\n\n";
    std::cout << "Theoretical Bandwith:        "
              << yacx::theoretical_bandwidth(dev) << " GB/s\n";

    //2B. Set kernel string and compile options
    Source source{load("examples/kernels/sum_Array.cu")};

    Options options;
    options.insert("--std", "c++14");
    options.insert("--device-debug");
    options.insertOptions(yacx::options::GpuArchitecture{dev});
    options.insertOptions(yacx::options::FMAD(true));

    //2C. Set kernel arguments
    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{A, matrix_size});
    args.emplace_back(KernelArg{B, matrix_size});
    args.emplace_back(KernelArg{C_cuda, matrix_size,true,false});

    //2D. Compile Kernels
    dim3 block(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(2);

    Kernel kernel_sumArray = source.program("sumArrayOnGPU").instantiate(type_of(data)).
    compile(options).configure(grid,block);

    //2E. Launch kernels

    //CPU single threaded sum array
    start = std::clock();
    sumArrayOnHost<float>(A,B,C_seq,NX,NY,NZ);
    std::cout << "Time\u001b[33m[CPU single threaded]\u001b[0m:   "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

    time = kernel_sumArray.launch(args);
    std::cout << "Time\u001b[33m[sumArrayOnGPU]\u001b[0m:      "
              << time.total << " ms\n";

    std::cout << "Effective Bandwith:          "
              << yacx::effective_bandwidth(time.launch,  args.size()) << " GB/s\n";

    equalSumArray = std::equal(C_cuda,C_cuda+(NX+NY+NY),C_seq,comparator<float>);

    if (!equalSumArray)
      compare<float>(C_seq,C_cuda,NX+NY+NZ);

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  if (equalSumArray) {
    std::cout << "\u001b[32meverything was correctly calculated!\u001b[0m\n";
    std::cout << "upload time:     " << time.upload
              << " ms\nexecution time:  " << time.launch
              << " ms\ndownload time    " << time.download
              << " ms\ntotal time:      " << time.total << " ms.\n";
  } else {
    std::cout << "\u001b[31m";
  }

  if (!equalSumArray) {
    std::cout << "SumArray went wrong ;_;\n";
  }

  std::cout << "\u001b[0m===================================" << std::endl;

  // Free resources

  delete[] A;
  delete[] B;
  delete[] C_seq;
  delete[] C_cuda;

  return 0;
}