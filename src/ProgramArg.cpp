#include "../include/cudaexecutor/ProgramArg.hpp"
#include "../include/cudaexecutor/Exception.hpp"

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result);                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::ProgramArg;

ProgramArg::ProgramArg(void *const data, size_t size, bool output)
    : _hdata{data}, _size{size}, _output{output} {}

void ProgramArg::upload() {
  CUDA_SAFE_CALL(cuMemAlloc(&_ddata, _size));
  CUDA_SAFE_CALL(cuMemcpyHtoD(_ddata, &_hdata, _size));
}

void ProgramArg::download() {
  if (_output)
    CUDA_SAFE_CALL(cuMemcpyDtoH(&_hdata, _ddata, _size));
  CUDA_SAFE_CALL(cuMemFree(_ddata));
}
