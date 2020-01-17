#pragma once

#include "JNIHandle.hpp"
#include <cuda.h>
#include <vector>

namespace yacx {

/*!
  \class ProgramArg ProgramArg.hpp
  \brief Class to help launch Kernel with given arguments
  Arguments are automatically uploaded and downloaded.
  \example docs/kernel_args.cpp
    Will execute the Kernel with <code>add(1, 2, 3, 4, result)</code>;
    and the result will be downloaded from device to host
*/

class KernelArg : JNIHandle {
  friend class KernelArgs;

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
  const void *content() const;
  //!
  //! \return pointer to device data
  CUdeviceptr deviceptr() { return m_ddata; }
  //! uploads data to device
  //! \return time to upload to device
  float upload();
  //! downloads data to host
  //! \return time to download from device
  float download() { return download(const_cast<void *>(m_hdata)); }
  //! downloads data to host
  //! \param pointer to host memory
  //! \return time to download from device
  float download(void *hdata);
  const size_t size() const { return m_size; }
  bool isDownload() const { return m_download; }
  void setDownload(bool download) { m_download = download; }
  bool isCopy() const { return m_copy; }
  void setCopy(bool copy) { m_copy = copy; }

 protected:
  //! copy data from host to device
  virtual void copyDataHtoD();
  //! copy data from device to host
  virtual void copyDataDtoH();
  const void *m_hdata;
  CUdeviceptr m_ddata;

 private:
  const size_t m_size;
  bool m_download;
  bool m_copy;
  const bool m_upload;
};

class KernelArgMatrixPadding : public KernelArg {
 public:
  //! A constructor
  /*!
   *
   * \param data pointer to argument for kernel function
   * \param size size of argument in bytes
   * \param dst_rows number of rows for new matrix with padding
   * \param dst_columns number of columns for new matrix with padding
   * \param src_rows number of rows of current matrix without padding
   * \param src_columns number of columns of currentmatrix without padding
   * \param paddingValue value to fill up additional rows and columns
   * \param shortElements true if size of elements in the matrix is 2 byte,
   * false if size of elements in the matrix is 4 byte
   * \param download copy the results from device to host after kernel execution
   * types, e.g. int)
   */
  KernelArgMatrixPadding(void *data, size_t size, int dst_rows, int dst_columns,
                         int src_rows, int src_columns, int paddingValue,
                         bool shortElements, bool download = false)
      : KernelArg(data, size, download, true, true),
        m_paddingValue(paddingValue), m_shortElements(shortElements),
        m_dst_rows(shortElements ? dst_rows * 2 : dst_rows * 4),
        m_dst_columns(shortElements ? dst_columns * 2 : dst_columns * 4),
        m_src_rows(shortElements ? src_rows * 2 : src_rows * 4),
        m_src_columns(shortElements ? src_columns * 2 : src_columns * 4) {}

 protected:
  void copyDataHtoD() override;
  void copyDataDtoH() override;

 private:
  const bool m_shortElements;
  const int m_paddingValue;
  const int m_dst_rows;
  const int m_dst_columns;
  const int m_src_rows;
  const int m_src_columns;
};

class KernelArgs {
 public:
  KernelArgs(std::vector<KernelArg> args);
  float upload();
  float download();
  float download(void *hdata);
  const void **content();
  size_t size() const;
  size_t maxOutputSize() const;

 private:
  std::vector<KernelArg> m_args;
  mutable std::vector<const void *> m_voArgs;
};
} // namespace yacx