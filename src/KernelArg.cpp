#include "yacx/Exception.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Logger.hpp"

#include <builtin_types.h>
#include <stdexcept>

using yacx::KernelArg, yacx::KernelArgMatrixPadding, yacx::loglevel,
    yacx::detail::DataCopy, yacx::detail::DataCopyKernelArg,
    yacx::detail::DataCopyKernelArgMatrixPadding;

std::shared_ptr<DataCopyKernelArg> KernelArg::dataCopyKernelArg =
    std::make_shared<DataCopyKernelArg>();

KernelArg::KernelArg(void *const data, size_t size, bool download, bool copy,
                     bool upload)
    : m_hdata{data}, m_dataCopy(KernelArg::dataCopyKernelArg), m_size{size},
      m_download{download}, m_copy{copy}, m_upload{upload} {
  logger(loglevel::DEBUG) << "created KernelArg with size: " << size
                          << ", which should " << (m_upload ? "be" : "not be")
                          << " uploaded and should "
                          << (m_download ? "be" : "not be") << " downloaded";
}

KernelArgMatrixPadding::KernelArgMatrixPadding(void *data, size_t size,
                                               bool download, int elementSize,
                                               unsigned int paddingValue,
                                               int src_rows, int src_columns,
                                               int dst_rows, int dst_columns)
    : KernelArg(data, size, download, true, true) {
  m_dataCopy = std::make_shared<DataCopyKernelArgMatrixPadding>(
      elementSize, paddingValue, src_rows, src_columns, dst_rows, dst_columns);

  logger(loglevel::DEBUG) << "created KernelArgMatrixPadding with src_rows: "
                          << src_rows << ", src_columns: " << src_columns
                          << ", dst_rows: " << dst_rows
                          << ", dst_columns: " << dst_columns
                          << ", paddingValue: " << paddingValue;
}

void KernelArg::upload(CUstream stream) {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "uploading argument";
    CUDA_SAFE_CALL(cuMemAlloc(&m_ddata, m_size));

    if (m_copy) {
      logger(loglevel::DEBUG1) << "copying data to device";

      m_dataCopy.get()->copyDataHtoD(this, stream);
    }
  } else {
    logger(loglevel::DEBUG1) << "NOT uploading argument";
  }
}

void KernelArg::download(void *hdata, CUstream stream) {
  if (m_download) {
    logger(loglevel::DEBUG1) << "downloading argument";

    m_dataCopy.get()->copyDataDtoH(this, stream);
  } else {
    logger(loglevel::DEBUG1) << "NOT downloading argument";
  }
}

void KernelArg::free(){
  if (m_upload) {
    logger(loglevel::DEBUG1) << "freeing argument from device";
    CUDA_SAFE_CALL(cuMemFree(m_ddata));
  } else {
    logger(loglevel::DEBUG1) << "NOT freeing argument from device";
  }
}

const void *KernelArg::content() const {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "returning device pointer";
    return &m_ddata;
  }
  logger(loglevel::DEBUG1) << "returning host pointer";
  return m_hdata;
}

void DataCopyKernelArg::copyDataHtoD(void *hdata, CUdeviceptr ddata,
                                     size_t size, CUstream stream) {
  CUDA_SAFE_CALL(cuMemcpyHtoDAsync(ddata, hdata, size, stream));
}

void DataCopyKernelArg::copyDataDtoH(CUdeviceptr ddata, void *hdata,
                                     size_t size, CUstream stream) {
  CUDA_SAFE_CALL(cuMemcpyDtoHAsync(hdata, ddata, size, stream));
}

void DataCopyKernelArgMatrixPadding::copyDataHtoD(void *hdata,
                                                  CUdeviceptr ddata, size_t, CUstream stream) {
  char *src = static_cast<char *>(hdata);

  const unsigned char paddingValueChar =
      m_paddingValue >> (sizeof(int) - sizeof(char));
  const unsigned short paddingValueShort =
      m_paddingValue >> (sizeof(int) - sizeof(short));

  switch (m_elementSize) {
  case 1:
    CUDA_SAFE_CALL(cuMemsetD8Async(ddata, paddingValueChar,
                                   m_dst_rows * m_dst_columns, stream));
    break;
  case 2:
    CUDA_SAFE_CALL(cuMemsetD16Async(ddata, paddingValueShort,
                                    m_dst_rows * m_dst_columns, stream));
    break;
  case 4:
    CUDA_SAFE_CALL(cuMemsetD32Async(ddata, m_paddingValue,
                                    m_dst_rows * m_dst_columns, stream));
    break;
  default:
    throw std::invalid_argument("invalid elementsize of paddingArg. Only 1,2 "
                                "or 4 bytes elementsize are supported.");
  }

  const size_t sizeSrcColumn = m_src_columns * m_elementSize;
  const size_t sizeDstColumn = m_dst_columns * m_elementSize;

  for (int i = 0; i < m_src_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyHtoDAsync(ddata, src, sizeSrcColumn, stream));

    ddata += sizeDstColumn;
    src += sizeSrcColumn;
  }
}

void DataCopyKernelArgMatrixPadding::copyDataDtoH(CUdeviceptr ddata,
                                                  void *hdata, size_t, CUstream stream) {
  char *src = static_cast<char *>(hdata);

  const size_t sizeSrcColumn = m_src_columns * m_elementSize;

  for (int i = 0; i < m_src_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyDtoHAsync(src, ddata, sizeSrcColumn, stream));

    ddata += m_dst_columns * m_elementSize;
    src += sizeSrcColumn;
  }
}