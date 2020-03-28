#include "kernels/gauss.h"
#include "yacx/main.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

#define NUM_THREADS 16
#define NUM_BLOCKS 32

using yacx::Source, yacx::KernelArg, yacx::Kernel, yacx::Options, yacx::Device,
    yacx::load, yacx::type_of, yacx::Headers, yacx::Header;

void writePPM(Pixel *pixels, const char *filename, int width, int height) {
  std::ofstream outputFile(filename, std::ios::binary);

  // write header:
  outputFile << "P6\n" << width << " " << height << "\n255\n";

  outputFile.write(reinterpret_cast<const char *>(pixels),
                   sizeof(Pixel) * width * height);
}

// Pointer returned must be explicitly freed!
Pixel *readPPM(const char *filename, int *width, int *height) {
  std::ifstream inputFile(filename, std::ios::binary);

  // parse harder
  // first line: P6\n
  inputFile.ignore(2, '\n'); // ignore P6
  // possible comments:
  while (inputFile.peek() == '#') {
    inputFile.ignore(1024, '\n');
  } // skip comment
  // next line: width_height\n
  inputFile >> (*width);
  inputFile.ignore(1, ' '); // ignore space
  inputFile >> (*height);
  inputFile.ignore(1, '\n'); // ignore newline
  // possible comments:
  while (inputFile.peek() == '#') {
    inputFile.ignore(1024, '\n');
  } // skip comment
  // last header line: 255\n:
  inputFile.ignore(3, '\n'); // ignore 255 and newline

  Pixel *data = new Pixel[(*width) * (*height)];

  inputFile.read(reinterpret_cast<char *>(data),
                 sizeof(Pixel) * (*width) * (*height));

  return data;
}

void calculateWeights(float weights[5][5]) {
  float sigma = 1.0;
  float r, s = 2.0 * sigma * sigma;

  // sum is for normalization
  float sum = 0.0;

  // generate weights for 5x5 kernel
  for (int x = -2; x <= 2; x++) {
    for (int y = -2; y <= 2; y++) {
      r = x * x + y * y;
      weights[x + 2][y + 2] = exp(-(r / s)) / (M_PI * s);
      sum += weights[x + 2][y + 2];
    }
  }

  // normalize the weights
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      weights[i][j] /= sum;
    }
  }
}

int main(int argc, char **argv) {
  const char *inFilename = (argc > 1) ? argv[1] : "kernels/lena.ppm";
  const char *outFilename = (argc > 2) ? argv[2] : "kernels/output.ppm";

  float weights[5][5];
  calculateWeights(weights);
  int width;
  int height;

  Pixel *image = readPPM(inFilename, &width, &height);

  try {
    Source source{load("examples/kernels/gauss.cu")};

    size_t size_pixel = height * width * sizeof(Pixel);
    size_t size_weights = 5 * 5 * sizeof(float);

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{image, size_pixel, true});
    args.emplace_back(KernelArg{weights, size_weights});
    args.emplace_back(KernelArg{&width});
    args.emplace_back(KernelArg{&height});

    dim3 block;
    dim3 grid(width, height);
    source.program("gaussFilterKernel")
        .compile()
        .configure(grid, block)
        .launch(args);
  } catch (const std::exception &e) {
    std::cerr << "Error:\n" << e.what() << std::endl;
  }

  writePPM(image, outFilename, width, height);
  free(image);

  return 0;
}
