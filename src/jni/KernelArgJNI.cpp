#include "KernelArgJNI.hpp"

#include <cstring>
#include <stdio.h>

using jni::KernelArgJNI, cudaexecutor::KernelArg;

KernelArgJNI::KernelArgJNI(void* const data, size_t size, bool download, bool copy, bool upload) {
    _hdata = malloc(size);
    if (data)
        std::memcpy(_hdata, data, size);

    kernelArg = new KernelArg{_hdata, size, download, copy, upload};
}

KernelArgJNI::~KernelArgJNI() {
    free(_hdata);
    delete kernelArg;
}