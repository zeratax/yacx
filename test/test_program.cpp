//Here, the programs of kernels are tested.
#include "yacx/Headers.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Source.hpp"
#include "yacx/Program.hpp"

#include <catch2/catch.hpp>
#include <iostream>

using yacx::Program, yacx::KernelArg, yacx::Source, yacx::Header, yacx::Headers;

TEST_CASE("A program is created, compiled, configured and then launched."){
    //A. Preparing the input for the kernel-compilation using source
    int datasize{10};
    int *hX = new int[10]{1,2,3,4,5,6,7,8,9,10};
    int *hY = new int[10]{6,7,8,9,10,11,12,13,14,15};
    int *hOut = new int[10];
    size_t bufferSize = 10*sizeof(int);
    
    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{hX, bufferSize});
    args.emplace_back(KernelArg{hY, bufferSize});
    args.emplace_back(KernelArg{hOut, bufferSize, true});
    args.emplace_back(KernelArg(&datasize)); 
    
    Headers headers;
    headers.insert(Header{"cuda_runtime.h"});

    //B. Preparing the output for kernel-compilation
    int *hostCompareOutput = new int[10]{7,9,11,13,15,17,19,21,23,25};

   //C. Creating, Compiling, Configuring and Launching a program.
    Source source{
        "#include \"cuda_runtime.h\"\n"
        "extern \"C\"\n"
        "__global__ void cuda_add(int *x, int *y, int *out, int "
        "datasize) {\n"
        " int i = threadIdx.x;\n"
        " out[i] = x[i] + y[i];\n"
        "}", headers};
        
    dim3 grid_test(1);
    dim3 block_test(10);
    
    source.program("cuda_add").compile().configure(grid_test, block_test).launch(args);
        
    //D. Comparing the results
    for (int i=0;i<10;i++){
        REQUIRE(hOut[i] == hostCompareOutput[i]);
    }

    delete[] hX;
    delete[] hY;
    delete[] hOut;
    delete[] hostCompareOutput;
}
