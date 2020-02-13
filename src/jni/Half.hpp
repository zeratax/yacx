#pragma once

#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/Program.hpp"
#include "../../include/yacx/Source.hpp"

#include <stdlib.h>

void convertFtoH(void *floats, void *halfs, unsigned int length);

void convertHtoF(void *halfs, void *floats, unsigned int length);

void initKernel();