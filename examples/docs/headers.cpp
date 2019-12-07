#include "yacx/Header.hpp"
#include "yacx/Headers.hpp"
#include "yacx/Source.hpp"
#include "kernels/gauss.h"

using yacx::Source, yacx::Headers, yacx::Header;

Headers headers;
headers.insert(Header{"kernels/gauss.h"});
Source source{load("kernels/gauss.cu"), headers};

// Alternatively if you only use one header

Source source{load("kernels/gauss.cu"), Headers{"kernels/gauss.h"}};