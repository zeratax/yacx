#include "Half.hpp"

void convertFtoH(void* floats, void* halfs, unsigned int sizeFloats){
    int* data = static_cast<int*> (floats);
    short* dataHalf = static_cast<short*> (halfs);

    for (unsigned int i = 0, j = 0; i < sizeFloats/4; i++, j++) {
        int float32 = data[i];
        short float16 = ((float32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
        float16 |= ((float32 & 0x80000000) >> 16);
        dataHalf[j] = float16;
    }
}

void convertHtoF(void* floats, void* halfs, unsigned int sizeHalfs){
    //TODO
}