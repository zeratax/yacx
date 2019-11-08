#ifndef CUDAEXECUTOR_PROGRAMARG_HPP_
#define CUDAEXECUTOR_PROGRAMARG_HPP_

#include <cuda.h>

namespace cudaexecutor {
class ProgramArg {
  void *_hdata;
  CUdeviceptr _ddata;
  bool _output;
  size_t _size;

 public:
  ProgramArg(void *data, size_t size, bool output = false);
  void *content() const { return _hdata; }
  void upload();
  void download();
};
} // namespace cudaexecutor
#endif // CUDAEXECUTOR_PROGRAMARG_HPP_
