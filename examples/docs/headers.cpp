#include "cudaexecutor/Header.hpp"
#include "cudaexecutor/Headers.hpp"
#include "cudaexecutor/Source.hpp"
#include "kernels/gauss.h"

using cudaexecutor::Source, cudaexecutor::Headers, cudaexecutor::Header;

Headers headers;
headers.insert(Header{"kernels/gauss.h"});
Source source{load("kernels/gauss.cu"), headers};

// Alternatively if you only use one header

Source source{load("kernels/gauss.cu"), Headers{"kernels/gauss.h"}};