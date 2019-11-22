#pragma once

#include <cuda.h>

namespace cudaexecutor {
class ProgramArg {
  const void *_hdata;
  CUdeviceptr _ddata;
  const bool _download;
  const bool _upload;
  const bool _copy;
  size_t _size;

 public:
  ProgramArg(void *data, size_t size, bool download = false, bool copy = true,
             bool upload = true);
  ProgramArg(void *data) : ProgramArg{data, 0, false, false, false} {};
  const void *content();
  void upload();
  void download();
  CUdeviceptr deviceptr() { return _ddata; }
};
} // namespace cudaexecutor
