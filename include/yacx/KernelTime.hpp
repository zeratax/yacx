#pragma once
#include "Devices.hpp"
#include "KernelArgs.hpp"

#include <iostream>

namespace yacx {
float effective_bandwidth(float miliseconds, size_t sizeInBytes);
float theoretical_bandwidth(Device &device);

typedef struct KernelTimeStruct {
  float upload{0};
  float download{0};
  float launch{0};
  float total{0};
  size_t size_download{0};
  size_t size_upload{0};
  size_t size_total{0};

  float effective_bandwidth_up() {
    return effective_bandwidth(upload, size_upload);
  }
  float effective_bandwidth_down() {
    return effective_bandwidth(download, size_download);
  }
  float effective_bandwidth_launch() {
    return effective_bandwidth(launch, size_total);
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const struct KernelTimeStruct &time) {
    os << "upload time:     " << time.upload
       << " ms\nexecution time:  " << time.launch << " ms\ndownload time    "
       << time.download << " ms\ntotal time:      " << time.total << " ms.\n";
    return os;
  }
} KernelTime;
} // namespace yacx
