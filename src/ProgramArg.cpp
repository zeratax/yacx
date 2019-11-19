#include "../include/cudaexecutor/ProgramArg.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Logger.hpp"

using cudaexecutor::ProgramArg, cudaexecutor::loglevel;

ProgramArg::ProgramArg(void *const data, size_t size, bool download,
                       bool upload)
    : _hdata{data}, _size{size}, _download{download}, _upload{upload} {
  logger(loglevel::DEBUG) << "created ProgramArg with size: " << size
                          << ", which should " << (_upload ? "be" : "not be")
                          << " uploaded and should "
                          << (_download ? "be" : "not be") << " downloaded";
}

void ProgramArg::upload() {
  if (_upload) {
    CUDA_SAFE_CALL(cuMemAlloc(&_ddata, _size));
    // memcopy doesn't always have to be called
    CUDA_SAFE_CALL(cuMemcpyHtoD(_ddata, &_hdata, _size));
  }
}

void ProgramArg::download() {
  if (_download)
    CUDA_SAFE_CALL(cuMemcpyDtoH(&_hdata, _ddata, _size));
  CUDA_SAFE_CALL(cuMemFree(_ddata));
}

void *ProgramArg::content() {
  if (_upload)
    return _hdata;
  return &_ddata;
}