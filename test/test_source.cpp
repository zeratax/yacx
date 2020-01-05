//Here, the sources of kernels are tested.
#include "yacx/Headers.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Source.hpp"

#include <catch2/catch.hpp>
#include <iostream>

using yacx::KernelArg, yacx::Source, yacx::Header, yacx::Headers;

TEST_CASE("The source of a kernel is created and then through it a program is created."){
        //A. Preparing the input for the kernel-compilation using source
        int datasize{5};
        int *hX = new int[5]{1,2,3,4,5};
        int *hY = new int[5]{6,7,8,9,10};
        int *hOut = new int[5];
        size_t bufferSize = 5*sizeof(int);

        std::vector<KernelArg> args;
        args.emplace_back(KernelArg{hX, bufferSize});
        args.emplace_back(KernelArg{hY, bufferSize});
        args.emplace_back(KernelArg{hOut, bufferSize, true});
        args.emplace_back(KernelArg(&datasize));

        Headers headers;
        headers.insert(Header{"cuda_runtime.h"});

        //A2. Preparing the output for kernel-compilation
        int *hostCompareOutput = new int[5]{7,9,11,13,15};
        
        Source source{
            "#include \"cuda_runtime.h\"\n"
            "extern \"C\"\n"
            "__global__ void cuda_add(int *x, int *y, int *out, int "
            "datasize) {\n"
            " int i = threadIdx.x;\n"
            " out[i] = x[i] + y[i];\n"
            "}", headers};
            
        dim3 grid(1);
        dim3 block(5);
        
        //B. Compilation of Kernels through the creation of a 
        //program from kernel - source
        source.program("cuda_add").compile().configure(grid, block).launch(args);

        //C. Comparing the results
        for (int i=0;i<5;i++){
            REQUIRE(hOut[i] == hostCompareOutput[i]);
        }
    
}
