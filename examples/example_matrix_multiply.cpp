#include "cudaexecutor/main.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::Kernel;

void compare(float *lhs, float *rhs, int width) {
  int errors = 0;
  for (int i{0}; i < width; i += 1) {
    // printf("[%d] expected %f : actually %f\n", i, lhs[i], rhs[i]);
    if ((lhs[i] - rhs[i]) != 0) {
      errors += 1;
    }
  }
  if (errors > 0)
    printf("%d errors occured, out of %d values.\n", errors, width);
  else
    printf("no errors occured.\n");
}

std::function<bool(const float &, const float &)> comparator =
    [](const float &left, const float &right) {
      // double epsilon{1.0E-8};
      double epsilon{1};
      return (abs(left - right) < epsilon);
    };

template <class Iter>
void fill(Iter start, Iter end, int min = 0, int max = 100) {
  static std::random_device rd;
  static std::mt19937 mte(rd());

  std::uniform_int_distribution<int> dist(min, max);

  std::generate(start, end, [&]() { return dist(mte); });
}

void MatrixMulSeq(const float *M, const float *N, float *P, size_t width) {
  size_t Col, Row, k;
  for (Col = 0; Col < width; ++Col)
    for (Row = 0; Row < width; ++Row) {
      float sum = 0;
      for (k = 0; k < width; k += 1) {
        sum += M[Row * width + k] * N[k * width + Col];
      }
      P[Row * width + Col] = sum;
    }
}

int main() {
  std::clock_t start;

  bool equalMultiply1, equalMultiply1unfolded, equalMultiply2,
      equalMultiplyNaive;

  const size_t WIDTH{1024};
  const size_t BLOCK_SIZE{16};
  const size_t GRANULARITY{4};
  const size_t matrix_size{WIDTH * WIDTH * sizeof(float)};
  static_assert(WIDTH % BLOCK_SIZE == 0);
  static_assert(BLOCK_SIZE % GRANULARITY == 0);
  // problem with WIDTH > 500
  // std::array<float, WIDTH * WIDTH> M, N, P_cuda, P_seq;
  float *M = new float[WIDTH * WIDTH];
  float *N = new float[WIDTH * WIDTH];
  float *P_seq = new float[WIDTH * WIDTH];
  float *P_cuda = new float[WIDTH * WIDTH];

  // Fill arrays with random values
  // fill(N.begin(), N.end());
  // fill(M.begin(), M.end());
  fill(M, M + (WIDTH * WIDTH));
  fill(N, N + (WIDTH * WIDTH));

  try {
    // Select Device
    Device dev;
    std::cout << "Selected " << dev.name() << " with "
              << (dev.total_memory() / 1024) / 1024 << "mb VRAM" << std::endl;
    std::cout << "Arguments have a combined size of "
              << ((matrix_size * 3 + sizeof(size_t)) / 1024) << "kb"
              << std::endl;

    // Set kernel string and compile options

    Source source{load("kernels/matrixMult.cu")};
    Options options;
    options.insert("--std", "c++14");
    options.insert("--device-debug");
    options.insertOptions(cudaexecutor::options::GpuArchitecture{dev});

    // Set arguments

    std::vector<ProgramArg> args;
    // args.emplace_back(ProgramArg{M.data(), matrix_size});
    // args.emplace_back(ProgramArg{N.data(), matrix_size});
    // args.emplace_back(ProgramArg{P_cuda.data(), matrix_size, true, false});
    args.emplace_back(ProgramArg{M, matrix_size});
    args.emplace_back(ProgramArg{N, matrix_size});
    args.emplace_back(ProgramArg{P_cuda, matrix_size, true, false});
    args.emplace_back(ProgramArg{const_cast<size_t *>(&WIDTH)});

    // Compile Kernels

    dim3 grid(WIDTH, WIDTH);
    dim3 block(1, 1);
    Kernel kernelNaive = source.program("MatrixMultyNaive")
                             .compile(options)
                             .configure(grid, block);

    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE / GRANULARITY;
    grid.x = WIDTH / block.x;
    grid.y = WIDTH / GRANULARITY / block.y;
    //    std::cout << "get_global_size(0): " << block.x * grid.x << std::endl;
    //    std::cout << "get_global_size(1): " << block.y * grid.y << std::endl;
    //    std::cout << "get_local_size(0): " << block.x << std::endl;
    //    std::cout << "get_local_size(1): " << block.y << std::endl;
    Kernel kernel1 = source.program("MatrixMulty1")
                         .instantiate(BLOCK_SIZE, GRANULARITY)
                         .compile(options)
                         .configure(grid, block);

    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE / 4;
    grid.x = WIDTH / block.x;
    grid.y = WIDTH / 4 / block.y;
    Kernel kernel1_1 = source.program("MatrixMulty1unfolded")
                           .instantiate(BLOCK_SIZE)
                           .compile(options)
                           .configure(grid, block);

    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    grid.x = WIDTH / BLOCK_SIZE;
    grid.y = WIDTH / BLOCK_SIZE;
    Kernel kernel2 = source.program("MatrixMulty2")
                         .instantiate(BLOCK_SIZE)
                         .compile(options)
                         .configure(grid, block);

    // Launch kernels

    // CPU single threaded matrix multiplication
    start = std::clock();
    // MatrixMulSeq(M.data(), N.data(), P_seq.data(), WIDTH);
    MatrixMulSeq(M, N, P_seq, WIDTH);
    std::cout << "Time[CPU single threaded]: "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

    start = std::clock();
    kernelNaive.launch(args, dev);
    std::cout << "Time[MatrixMultyNaive]: "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

    equalMultiplyNaive =
        std::equal(P_cuda, P_cuda + (WIDTH * WIDTH), P_seq, comparator);

    if (BLOCK_SIZE % 4 == 0) {
      start = std::clock();
      kernel1_1.launch(args, dev);
      std::cout << "Time[MatrixMulty1unfolded]: "
                << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
                << " ms" << std::endl;
      equalMultiply1unfolded =
          std::equal(P_cuda, P_cuda + (WIDTH * WIDTH), P_seq, comparator);
    } else {
      equalMultiply1unfolded = true;
    }

    start = std::clock();
    kernel1.launch(args, dev);
    std::cout << "Time[MatrixMulty1]: "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;
    compare(P_seq, P_cuda, WIDTH * WIDTH);
    equalMultiply1 =
        std::equal(P_cuda, P_cuda + (WIDTH * WIDTH), P_seq, comparator);

    start = std::clock();
    kernel2.launch(args, dev);
    std::cout << "Time[MatrixMulty2]: "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

    equalMultiply2 =
        std::equal(P_cuda, P_cuda + (WIDTH * WIDTH), P_seq, comparator);

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  if (equalMultiplyNaive && equalMultiply1 && equalMultiply1unfolded &&
      equalMultiply2) {
    std::cout << "everything was correctly calculated!" << std::endl;
  }
  if (!equalMultiplyNaive) {
    std::cout << "Naive went wrong ;_;" << std::endl;
  }
  if (!equalMultiply1) {
    std::cout << "Multy1 went wrong ;_;" << std::endl;
  }
  if (!equalMultiply1unfolded) {
    std::cout << "Multy1unfolded went wrong ;_;" << std::endl;
  }
  if (!equalMultiply2) {
    std::cout << "Multy2 went wrong ;_;" << std::endl;
  }

  // Free resources

  delete[] M;
  delete[] N;
  delete[] P_seq;
  delete[] P_cuda;

  return 0;
}