#pragma once

#include <cuda.h>

namespace cudaexecutor {
/*!
  \class ProgramArg ProgramArg.hpp
  \brief Class to help launch Kernel with given arguments
  Arguments are automatically uploaded and downloaded.
  \example docs/kernel_args.cpp
    Will execute the Kernel with <code>add(1, 2, 3, 4, result)</code>;
    and the result will be downloaded from device to host
*/
class KernelArg {
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
  KernelArg(void *data, size_t size, bool download = false, bool copy = true,
             bool upload = true);
  //! A constructor for basic types, e.g. int
  //! \param data pointer to argument for kernel function
  explicit KernelArg(void *data) : KernelArg{data, 0, false, false, false} {};
  //!
  //! \return pointer to host data
  const void *content();
  //!
  //! \return pointer to device data
  CUdeviceptr deviceptr() { return m_ddata; }
  //! uploads data to device
  void upload();
  //! downloads data to host
  void download();
  const size_t size() { return m_size; }

 private:
  const void *m_hdata;
  const size_t m_size;
  CUdeviceptr m_ddata;
  const bool m_download;
  const bool m_copy;
  const bool m_upload;
};
} // namespace cudaexecutor
