#ifndef CUDAEXECUTOR_PROGRAMARG_HPP_
#define CUDAEXECUTOR_PROGRAMARG_HPP_

#include <cuda.h>

namespace cudaexecutor {
class ProgramArg {
  void *_hdata;
  CUdeviceptr _ddata;
  bool _download, _upload;
  size_t _size;

 public:
  ProgramArg(void *data, size_t size, bool download = false, bool upload = true);
  void *content();
  void upload();
  void download();
};
} // namespace cudaexecutor
#endif // CUDAEXECUTOR_PROGRAMARG_HPP_
