#pragma once

#include <stdlib.h>

void convertFtoH(void *floats, void *halfs, unsigned int sizeFloats);

void convertHtoF(void *halfs, void *floats, unsigned int sizeHalfs);
