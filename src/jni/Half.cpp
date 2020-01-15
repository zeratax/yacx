#include "Half.hpp"

void convertFtoH(void* floats, void* halfs, unsigned int sizeFloats){
    int *data = static_cast<int*> (floats);
    short *dataHalf = static_cast<short*> (halfs);

    for (unsigned int i = 0, j = 0; i < sizeFloats/4; i++, j++) {
        int float32 = data[i];
        short float16 = ((float32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
        float16 |= ((float32 & 0x80000000) >> 16);
        dataHalf[j] = float16;
    }
}

void convertHtoF(void* halfs, void* floats, unsigned int sizeHalfs){

  int *data = static_cast<int*> (floats);
  short *dataHalf = static_cast<short*> (halfs);

  short float16 = 0;
  int float32 = 0x00000000;
  for (unsigned int i = 0, j = 0; i < sizeHalfs/2 ; i++, j++) {

    float16 = dataHalf[i];
    float32 = 0x00000000;
    float32 |= ((float16 & 0xf800) << 16);

    if((float16 & 0x4000) == 0x4000) {
      float32 |= 0x40000000;
    } else {
      float32 |= 0x38000000;
    }
    float32 |= (float16 & 0x3C00) << 13;

    float32 |= (float16 & 0x3ff) << 13;

    data[j] = float32;

  }
}
