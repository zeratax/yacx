#pragma once

#include <cuda.h>

namespace cudaexecutor {
/*!
  \class ProgramArg ProgramArg.hpp
  \brief Class to help launch Kernel with given arguments
  Arguments are automatically uploaded and downloaded.
  \example program_args.cpp
    Will execute the Kernel with <code>add(1, 2, 3, 4, result)</code>;
    and the result will be downloaded from device to host
*/
class ProgramArg {
 public:
  //! A constructor
  /*!
   *
   * \param data pointer to argument for kernel function
   * \param size size of argument in bytes
   * \param download copy the results from device to host after kernel execution
   * \param copy copy the results to the device
   * \param upload allocate the argument on the device (not necessary for basic
   * types, e.g. int)
   */
  ProgramArg(void *data, size_t size, bool download = false, bool copy = true,
             bool upload = true);
  //! A constructor for basic types, e.g. int
  //! \param data pointer to argument for kernel function
  explicit ProgramArg(void *data) : ProgramArg{data, 0, false, false, false} {};
  //!
  //! \return pointer to host data
  const void *content();
  //!
  //! \return pointer to device data
  CUdeviceptr deviceptr() { return _ddata; }
  //! uploads data to device
  void upload();
  //! downloads data to host
  void download();
  const size_t size() { return _size; }

 private:
  const void *_hdata;
  CUdeviceptr _ddata;
  const bool _download;
  const bool _upload;
  const bool _copy;
  size_t _size;
};
} // namespace cudaexecutor
