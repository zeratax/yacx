//Here, the sources of kernels are tested.
#include "yacx/Headers.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Source.hpp"
#include "yacx/Exception.hpp"

#include "test_compare.hpp"
#include <catch2/catch.hpp>
#include <iostream>
#include <string>

using yacx::KernelArg, yacx::Source, yacx::Header, yacx::Headers;
using namespace std;

SCENARIO("Various sources of kernels are created and through them are created"
         ", then tested under the following conditions:"){
    GIVEN("Various sources of kernels with one or more functions."){
        //A1. Preparing the input for the kernel-compilation using various sources
        //Controlled results
        int datasize{5};
        int *hX = new int[5]{1,2,3,4,5};
        int *hY = new int[5]{6,7,8,9,10};
        int *hOut = new int[5];
        int *hOut_Multiply1 = new int[5];
        int *hOut_Multiply2 = new int[5];
        size_t bufferSize = 5*sizeof(int);

        vector<KernelArg> args;
        args.emplace_back(KernelArg{hX, bufferSize});
        args.emplace_back(KernelArg{hY, bufferSize});
        args.emplace_back(KernelArg{hOut, bufferSize, true});
        args.emplace_back(KernelArg(&datasize));
    
        Headers headers_default;
        headers_default.insert(Header{"cuda_runtime.h"});

        //A2. Preparing the output for kernel-compilation
        int *hostCompareOutput_Controlled = new int[5]{7,9,11,13,15};
        int *hostCompareOutput_1 = new int[5]{12,14,16,18,20};
        int *hostCompareOutput_2 = new int[5]{6,14,24,36,50};

        Source source{
            "#include \"cuda_runtime.h\"\n"
            "extern \"C\"\n"
            "__global__ void cuda_add(int *x, int *y, int *out, int "
            "datasize) {\n"
            " int i = threadIdx.x;\n"
            " out[i] = x[i] + y[i];\n"
            "}", headers_default};
            
        dim3 grid(1);
        dim3 block(5);
        
        /*A3. Compilation of Kernels through the creation of a 
          program from kernel - source*/
        source.program("cuda_add").compile().configure(grid, block).launch(args);

        //A4. Comparing the results
        for (int i=0;i<5;i++){
            REQUIRE(hOut[i] == hostCompareOutput_Controlled[i]);
        }

        //B1. Use of more than one header, in which checking-mechanisms are simulated.
        WHEN("More than one headers are used."){
            Headers headersNew1;
            headersNew1.insert(Header{"cuda_runtime.h"});
            headersNew1.insert(Header{"test_compare.hpp"});
           
            /*B2. Compilation of Kernels through the creation of a 
            program from kernel - source*/
            THEN("The results are consistent with the given controlled output."){
                Source sourceNew1{
                    "#include \"cuda_runtime.h\"\n"
                    "#include \"test_compare.hpp\"\n"
                    "extern \"C\"\n"
                    "__global__ void cuda_add_with_header(int *x, int *y, int *out, int"
                    " datasize) {\n"
                    " compare check = OUT_COMPARE_WRONG; \n"
                    " int i = threadIdx.x;\n"
                    " out[i] = x[i] + y[i] + 5;\n"
                    " if(out[i]%5==0) check = CORRECT;\n"
                    "}", headersNew1};

                dim3 gridNew1(1);
                dim3 blockNew1(5);
                
                sourceNew1.program("cuda_add_with_header").compile().configure(gridNew1, blockNew1).launch(args);
                
                //B3. Comparing the results
                for (int i=0;i<5;i++) REQUIRE(hOut[i] == hostCompareOutput_1[i]);
            }    
        }

        /**
         * C1. Checking for consistencies for running various functions in a kernel-code
         */
        WHEN("More one functions in a kernel-code are used."){
            Headers headersNew2;
            headersNew2.insert(Header{"cuda_runtime.h"});

            //C2. Compilation of Kernels through the creation of a program from kernel - source
            THEN("The results are consistent with the given controlled output."){
                /*
                * C3. Reusing the kernel-arguments arrays for testing of the coming functions in a kernel-code by
                * clearing the kernel-arguments array
                */
               args.clear(); 
               
               args.emplace_back(KernelArg{hX, bufferSize});
               args.emplace_back(KernelArg{hY, bufferSize});
               args.emplace_back(KernelArg{hOut_Multiply1, bufferSize, true});
               args.emplace_back(KernelArg(&datasize));
               
               Source sourceNew2{
                    "#include \"cuda_runtime.h\"\n"
                    "extern \"C\"\n"
                    "__global__ void cuda_add_normal(int *x, int *y, int *out, int "
                    "datasize) {\n"
                    " int i = threadIdx.x;\n"
                    " out[i] = x[i] + y[i];\n"
                    "}\n"
                    "extern \"C\"\n"
                    "__global__ void cuda_add_five(int *x, int *y, int *out, int"
                    " datasize) {\n"
                    " int i = threadIdx.x;\n"
                    " out[i] = x[i] + y[i] + 5;\n"
                    "}\n", headersNew2};

                dim3 gridNew2(1);
                dim3 blockNew2(5);
                
                sourceNew2.program("cuda_add_five").compile().configure(gridNew2, blockNew2).launch(args);
                
                //C4. Compiling another function in a kernel-code.
                /*
                * C5. Reusing the kernel-arguments arrays for testing of the coming functions in a kernel-code by
                *     clearing the kernel-arguments array
                */
                args.clear();

                args.emplace_back(KernelArg{hX, bufferSize});
                args.emplace_back(KernelArg{hY, bufferSize});
                args.emplace_back(KernelArg{hOut_Multiply2, bufferSize, true});
                args.emplace_back(KernelArg(&datasize));   
               
                sourceNew2.program("cuda_add_normal").compile().configure(gridNew2, blockNew2).launch(args);

                //C6. Comparing the results for the use of multiple functions in a kernel-code
                for (int i=0;i<5;i++) REQUIRE(hOut_Multiply1[i] == hostCompareOutput_1[i]);
                for (int i=0;i<5;i++) REQUIRE(hOut_Multiply2[i] == hostCompareOutput_Controlled[i]);
            }    
        }
        
        //D1. Checking for the validation of declaring functions
        WHEN("The naming of the kernel-compilation function is not consistent to that of that function of a kernel-code."){
            /*
            * D2. Reusing the kernel-arguments arrays for testing of the coming functions in a kernel-code by
            * learing the kernel-arguments array
            */
            args.clear();

            args.emplace_back(KernelArg{hX, bufferSize});
            args.emplace_back(KernelArg{hY, bufferSize});
            args.emplace_back(KernelArg{hOut_Multiply1, bufferSize, true});
            args.emplace_back(KernelArg(&datasize));   
           
            Headers headersNew3;
            headersNew3.insert(Header{"cuda_runtime.h"});
            
            /*D3. Compilation of Kernels through the creation of a 
            program from kernel - source*/
            THEN("The results for naming functions in a kernel-code are consistent."){
                const CUresult error{CUDA_ERROR_NOT_FOUND};

                Source sourceNew3{
                    "#include \"cuda_runtime.h\"\n"
                    "extern \"C\"\n"
                    "__global__ void cuda_multiply(int *x, int *y, int *out, int" 
                    " datasize) {\n"
                    " int i = threadIdx.x;\n"
                    " out[i] = x[i]*y[i];\n"
                    "}\n", headersNew3};
                    
                dim3 gridNew3(1);
                dim3 blockNew3(5);

                /* 
                * D4. Checking the possibility of calling a kernel compilation function using
                *     kernel function conventions.
                */
                REQUIRE_THROWS_WITH(sourceNew3.program("this_function_does_not_exist").compile().configure(gridNew3, blockNew3).launch(args),Catch::Contains(yacx::detail::whichError(error)));
            }
        }

    delete[] hX;
    delete[] hY;
    delete[] hOut;
    delete[] hOut_Multiply1;
    delete[] hOut_Multiply2;
    delete[] hostCompareOutput_Controlled;
    delete[] hostCompareOutput_1;
    delete[] hostCompareOutput_2;
    }
}
