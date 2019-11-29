#include "cudaexecutor/main.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::Kernel;

std::function<bool(const float &, const float &)> comparator =
    [](const float &left, const float &right) {
      double epsilon{1.0E-8};
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

void print_matrices(float *matrix, const char *file_Name, int x_dim, int y_dim,
                    int dim) {
  std::ofstream outFile;
  outFile.open(file_Name);

  outFile << std::fixed;
  outFile << std::setprecision(2);

  for (int i = 0; i < x_dim; i++) {

    for (int j = 0; j < y_dim; j++) {
      outFile << matrix[i * dim + j] << " ";
    }
    outFile << std::endl;
  }
}

int main() {
  std::clock_t start;

  const size_t WIDTH = 64;
  const size_t BLOCK_SIZE = 16;
  const size_t matrix_size = WIDTH * WIDTH * sizeof(float);
  std::array<float, WIDTH * WIDTH> M, N, P_cuda, P_seq;

  fill(N.begin(), N.end());
  fill(M.begin(), M.end());

  try {
    Device dev;
    std::cout << "Selected " << dev.name() << " with "
              << (dev.total_memory() / 1024) / 1024 << "mb" << std::endl;
    std::cout << "Arguments have a combined size of "
              << ((matrix_size * 3 + sizeof(size_t)) / 1024) << "kb"
              << std::endl;

    Source source{load("kernels/matrixMult.cu")};
    Options options;
    options.insert("--std", "c++14");

    std::vector<ProgramArg> args;
    args.emplace_back(ProgramArg{M.data(), matrix_size});
    args.emplace_back(ProgramArg{N.data(), matrix_size});
    args.emplace_back(ProgramArg{P_cuda.data(), matrix_size, true, false});
    args.emplace_back(ProgramArg{const_cast<size_t *>(&WIDTH)});

    //dim3 block(16, 16 / 4);
    //dim3 grid(WIDTH, WIDTH / 4);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WIDTH / BLOCK_SIZE, WIDTH / BLOCK_SIZE);
    Kernel kernel = source
                        .program("multiply")
                        .instantiate(BLOCK_SIZE)
                        .compile(options)
                        .configure(grid, block);

    start = std::clock();
    kernel.launch(args, dev);
    std::cout << "Time: "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  start = std::clock();
  MatrixMulSeq(M.data(), N.data(), P_seq.data(), WIDTH);
  std::cout << "Time: "
            << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms"
            << std::endl;

  bool equal =
      std::equal(P_cuda.begin(), P_cuda.end(), P_seq.begin(), comparator);

  if (equal) {
    std::cout << "everything was correctly calculated!" << std::endl;
  } else {
    std::string filename{"p_cuda.txt"};
    std::cout << "something went wrong ;_;\n see output: '" << filename << "'"
              << std::endl;
    print_matrices(P_cuda.data(), filename.c_str(), WIDTH, WIDTH, 1);
  }

  return 0;
}
