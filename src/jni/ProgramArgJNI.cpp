#include "ProgramArgJNI.hpp"

#include <cstring>
#include <stdio.h>

using jni::ProgramArgJNI, cudaexecutor::ProgramArg;

ProgramArgJNI::ProgramArgJNI(void* const data, size_t size, bool download, bool copy, bool upload) {
    _hdata = malloc(size);
    if (data)
        std::memcpy(_hdata, data, size);

    programArg = new ProgramArg{_hdata, size, download, copy, upload};
}

ProgramArgJNI::~ProgramArgJNI() {
    free(_hdata);
    delete programArg;
}