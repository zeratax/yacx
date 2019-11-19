#include "../include/cudaexecutor/ProgramArg.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Logger.hpp"

using cudaexecutor::ProgramArg, cudaexecutor::loglevel;

ProgramArg::ProgramArg(void *const data, size_t size, bool download, bool copy,
                       bool upload)
    : _hdata{data}, _size{size}, _download{download}, _copy{_copy},
      _upload{upload} {
  logger(loglevel::DEBUG) << "created ProgramArg with size: " << size
                          << ", which should " << (_upload ? "be" : "not be")
                          << " uploaded and should "
                          << (_download ? "be" : "not be") << " downloaded";
}

void ProgramArg::upload() {
  if (_upload) {
    CUDA_SAFE_CALL(cuMemAlloc(&_ddata, _size));
    if (_copy)
      CUDA_SAFE_CALL(cuMemcpyHtoD(_ddata, _hdata, _size));
  }
}

void ProgramArg::download() {
  if (_download) {
    logger(loglevel::DEBUG) << "downloading argument";
    CUDA_SAFE_CALL(cuMemcpyDtoH(_hdata, _ddata, _size));
  } else {
    logger(loglevel::DEBUG) << "NOT downloading argument";
  }

  if (_upload) {
    logger(loglevel::DEBUG) << "freeing argument from device";
    CUDA_SAFE_CALL(cuMemFree(_ddata));
  } else {
    logger(loglevel::DEBUG) << "NOT freeing argument from device";
  }
}

void *ProgramArg::content() {
  if (_upload) {
    logger(loglevel::DEBUG) << "uploading argument";
    return &_ddata;
  }
  logger(loglevel::DEBUG) << "NOT uploading argument";
  return _hdata;
}