/**
 * Here, the creation and the compilation of a program souce are tested prior to the creation of 
 * a kernel source.
 */
#include "yacx/Headers.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Source.hpp"
#include "yacx/Program.hpp"
#include "yacx/Exception.hpp"

#include <catch2/catch.hpp>
#include <iostream>
#include <string>

using yacx::Program, yacx::KernelArg, yacx::Options, yacx::Source, yacx::Header, yacx::Headers;

SCENARIO("Various programs are created and compiled under the following conditions."){
    GIVEN("Options for the compilation of a kernel are important factors."){
        //A1. Preparing the input for the creation and compilation of a source program
        Options options1, options3, options4, options5;
        std::string options_valueStr;

        const nvrtcResult error{NVRTC_ERROR_INVALID_OPTION};
        std::string error_String{yacx::detail::whichError(error)};
        error_String.erase(0,2);

        Headers headers;
        headers.insert(Header{"cuda_runtime.h"});
        
        //A2. Using the same source program for all conditions of source program compilation using various options
        Source source1{
            "#include \"cuda_runtime.h\"\n"
            "extern \"C\"\n"
            "__global__ void cuda_add(int *x, int *y, int *out, int "
            "datasize) {\n"
            " int i = threadIdx.x;\n"
            " out[i] = x[i] + y[i];\n"
            "}", headers};

        //B1. Controlled conditions of program creation and compilation
        WHEN("A. Program is created under controlled conditions, ex. using default or equivalent GPU-architecture."){ 
            options1.insertOptions(yacx::options::GpuArchitecture(3, 0));
            options1.insertOptions(yacx::options::FMAD(true));
           
            //B2. Controlled results using ex. default or equivalent GPU - architecture.
            THEN("The program can be compiled succesfully."){
                REQUIRE_NOTHROW(source1.program("cuda_add").compile(options1));
            }
        }

        //C1. Creating a program and compiling it using null values of Options
        WHEN("B. Program is created using null value options"){ 
            //C2. Result using null options.
            THEN("The program can be compiled succesfully."){
                REQUIRE_NOTHROW(source1.program("cuda_add").compile());
            }
        }
            
        //D1. Creating a program and compiling it using incompatible Options values
        WHEN("C. Program is created using improper options values"){
            Options options2({yacx::options::GpuArchitecture(8000, 0),yacx::options::FMAD(true)});
           
            //D2. Result using incompatible Option values.
            THEN("The program cannot be compiled."){
                REQUIRE_THROWS_WITH(source1.program("cuda_add").compile(options2),Catch::Contains(error_String));
            }
        }

        /**E1. Creating and compiling a program using multiple options with various methods of insecting
         *     new option values.
         */
        WHEN("D. Program is created using various multiple options with various inserting methods."){
            options_valueStr = "-restrict";
            
            options3.insertOptions(yacx::options::GpuArchitecture(3, 0),yacx::options::FMAD(true));
            options3.insert("-std=c++14");
            options3.insert("-builtin-move-forward","true");
            options3.insert("-maxrregcount=6000","");
            options3.insert("-use_fast_math");
            options3.insert(options_valueStr);
            
            //E2. Result using various multiple options
            THEN("The program can be created and compiled."){
                REQUIRE_NOTHROW(source1.program("cuda_add").compile(options3));
            }
        }

        //F1. Creating and compiling a program using multiple and repeating options.
        WHEN("E. Program is created using various multiple and repeating options."){
            options4.insertOptions(yacx::options::GpuArchitecture(3, 0),yacx::options::FMAD(true));
            options4.insert("-std=c++14");
            options4.insert("-std=c++11");

            //F2. Result using various multiple and repeating options
            THEN("The program cannot be created and compiled."){
                REQUIRE_THROWS_WITH(source1.program("cuda_add").compile(options4),Catch::Contains(error_String));
            }
        }

        //G1. Creating and compiling a program using multiple and repeating options.
        WHEN("F. Program is created using incorrect option syntaxes"){
            options5.insertOptions(yacx::options::GpuArchitecture(3, 0),yacx::options::FMAD(true));
            options5.insert("-std=C_PLUS_PLUS_14");

            //G2. Result using various multiple options
            THEN("The program cannot be created and compiled."){
                REQUIRE_THROWS_WITH(source1.program("cuda_add").compile(options5),Catch::Contains(error_String));
            }
        }
    }
}