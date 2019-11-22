#pragma once

#include <cuda.h>

namespace cudaexecutor {
class ProgramArg {
  void *_hdata;
  CUdeviceptr _ddata;
  bool _download, _upload, _copy;
  size_t _size;

 public:
  ProgramArg(void *data, size_t size, bool download = false, bool copy = true,
             bool upload = true);
  ProgramArg(void *data) : ProgramArg{data, 0, false, false, false} {};
  void *content();
  void upload();
  void download();
  CUdeviceptr deviceptr() { return _ddata; }
};
} // namespace cudaexecutor
#endif // CUDAEXECUTOR_PROGRAMARG_HPP_
