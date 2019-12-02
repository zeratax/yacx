#include "cudaexecutor/Kernel.hpp"
#include "cudaexecutor/KernelArg.hpp"
#include "cudaexecutor/Program.hpp"
#include "cudaexecutor/Source.hpp"

#include <vector_types.h>

using cudaexecutor::KernelArg;

int a = 1;
int b = 2;
int c = 3;
int d = 4;
int result = 0;

std::vector<KernelArg> args;
args.emplace_back(KernelArg{&a});
args.emplace_back(KernelArg{&b});
args.emplace_back(KernelArg{&c});
args.emplace_back(KernelArg{&d});
args.emplace_back(KernelArg{&result, sizeof(int), true});

dim3 grid(1);
dim3 block(1);
source.program("add").compile().configure(grid, block).launch(args);