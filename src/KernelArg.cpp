#include "yacx/Exception.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Logger.hpp"

#include <builtin_types.h>

using yacx::KernelArg, yacx::KernelArgMatrixPadding, yacx::loglevel;

KernelArg::KernelArg(void *const data, size_t size, bool download, bool copy,
                     bool upload)
    : m_hdata{data}, m_size{size},
      m_download{download}, m_copy{copy}, m_upload{upload} {
  logger(loglevel::DEBUG) << "created KernelArg with size: " << size
                          << ", which should " << (m_upload ? "be" : "not be")
                          << " uploaded and should "
                          << (m_download ? "be" : "not be") << " downloaded";
}

float KernelArg::upload() {
  float time{0};

  if (m_upload) {
    logger(loglevel::DEBUG1) << "uploading argument";
    CUDA_SAFE_CALL(cuMemAlloc(&m_ddata, m_size));
    if (m_copy) {
      logger(loglevel::DEBUG1) << "copying data to device";

      cudaEvent_t start, stop;
      CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
      CUDA_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));

      CUDA_SAFE_CALL(cuEventRecord(start, 0));
      copyDataHtoD();
      CUDA_SAFE_CALL(cuEventRecord(stop, 0));

      CUDA_SAFE_CALL(cuEventSynchronize(stop));
      CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, stop));
    }
  } else {
    logger(loglevel::DEBUG1) << "NOT uploading argument";
  }

  return time;
}

float KernelArg::download(void *hdata) {
  float time{0};

  if (m_download) {
    logger(loglevel::DEBUG1) << "downloading argument";

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    CUDA_SAFE_CALL(cuEventRecord(start, 0));
    copyDataDtoH();
    CUDA_SAFE_CALL(cuEventRecord(stop, 0));

    CUDA_SAFE_CALL(cuEventSynchronize(stop));
    CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, stop));
  } else {
    logger(loglevel::DEBUG1) << "NOT downloading argument";
  }

  if (m_upload) {
    logger(loglevel::DEBUG1) << "freeing argument from device";
    CUDA_SAFE_CALL(cuMemFree(m_ddata));
  } else {
    logger(loglevel::DEBUG1) << "NOT freeing argument from device";
  }

  return time;
}

const void *KernelArg::content() const {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "returning device pointer";
    return &m_ddata;
  }
  logger(loglevel::DEBUG1) << "returning host pointer";
  return m_hdata;
}

void KernelArg::copyDataHtoD() {
  CUDA_SAFE_CALL(cuMemcpyHtoD(m_ddata, const_cast<void *>(m_hdata), m_size));
}

void KernelArg::copyDataDtoH() {
  CUDA_SAFE_CALL(cuMemcpyDtoH(const_cast<void *>(m_hdata), m_ddata, m_size));
}

void KernelArgMatrixPadding::copyDataHtoD() {
  CUdeviceptr dst = m_ddata;
  char *src = static_cast<char *>(const_cast<void *>(m_hdata));
  size_t memsetSize = m_shortElements ? (m_dst_columns - m_src_columns) / 2
                                      : (m_dst_columns - m_src_columns) / 4;

  for (int i = 0; i < m_src_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyHtoD(dst, src, m_src_columns));
    if (m_shortElements) {
      CUDA_SAFE_CALL(
          cuMemsetD16(dst + m_src_columns, m_paddingValue, memsetSize));
    } else {
      CUDA_SAFE_CALL(
          cuMemsetD32(dst + m_src_columns, m_paddingValue, memsetSize));
    }

    dst += m_dst_columns;
    src += m_src_columns;
  }

  if (m_shortElements) {
    CUDA_SAFE_CALL(cuMemsetD16(dst, m_paddingValue,
                               (m_dst_rows - m_src_rows) * m_dst_columns));
  } else {
    CUDA_SAFE_CALL(cuMemsetD32(dst, m_paddingValue,
                               (m_dst_rows - m_src_rows) * m_dst_columns));
  }
}

void KernelArgMatrixPadding::copyDataDtoH() {
  CUdeviceptr dst = m_ddata;
  char *src = static_cast<char *>(const_cast<void *>(m_hdata));

  for (int i = 0; i < m_dst_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyDtoH(src, dst, m_dst_columns));
    dst += m_dst_columns;
    src += m_src_columns;
  }
}