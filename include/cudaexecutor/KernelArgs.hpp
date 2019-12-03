#pragma once

#include <vector>
#include <cuda.h>

class KernelArgs {
 public:
  KernelArgs(std::vector<KernelArg> args);
  float upload();
  float download();
  void **content();

 private:
  std::vector<KernelArg> m_args;
  mutable std::vector<const char *> m_chArgs;
};