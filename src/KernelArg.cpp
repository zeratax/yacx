#include "yacx/Exception.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Logger.hpp"

#include <builtin_types.h>

using yacx::KernelArg, yacx::KernelArgMatrixPadding, yacx::loglevel, yacx::detail::DataCopyKernelArg,
yacx::detail::DataCopyKernelArgMatrixPadding;

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
      m_dataCopy.get()->copyDataHtoD();
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
    m_dataCopy.get()->copyDataDtoH();
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

void DataCopyKernelArg::copyDataHtoD() {
  CUDA_SAFE_CALL(cuMemcpyHtoD(m_kernelArg->m_ddata, const_cast<void *>(m_kernelArg->m_hdata),
    m_kernelArg->m_size));
}

void DataCopyKernelArg::copyDataDtoH() {
  CUDA_SAFE_CALL(cuMemcpyDtoH(const_cast<void *>(m_kernelArg->m_hdata), m_kernelArg->m_ddata,
    m_kernelArg->m_size));
}

void DataCopyKernelArgMatrixPadding::copyDataHtoD() {
  CUdeviceptr dst = m_kernelArg->m_ddata;
  char *src = static_cast<char *>(const_cast<void *>(m_kernelArg->m_hdata));
  size_t numberElementsMemset = m_dst_columns - m_src_columns;

  logger(loglevel::DEBUG1) << "CopyDataHtoD MatrixPadding with src_columns: " << m_src_columns
  << ", src_rows: " << m_src_rows << ", dst_columns: " << m_dst_columns << ", dst_rows: " << m_dst_rows
  << ", paddingValue: " << m_paddingValue;
  
  for (int i = 0; i < m_src_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyHtoD(dst, src, m_src_columns));
    
    switch (m_elementSize){
      case 1:
        CUDA_SAFE_CALL(cuMemsetD8(dst, m_paddingValue, numberElementsMemset);
        break;
      case 2:
        CUDA_SAFE_CALL(cuMemsetD16(dst, m_paddingValue, numberElementsMemset);
        break;
      case 4:
        CUDA_SAFE_CALL(cuMemsetD32(dst, m_paddingValue, numberElementsMemset);
        break;
      default:
        throw invalid_argument("invalid elementsize of paddingArg. Only 1,2 or 4 bytes elementsize are supported.");
    }

    dst += m_dst_columns;
    src += m_src_columns;
  }

  numberElementsMemset = (m_dst_rows - m_src_rows) * m_dst_columns);
  switch (m_elementSize){
    case 1:
      CUDA_SAFE_CALL(cuMemsetD8(dst, m_paddingValue, numberElementsMemset);
      break;
    case 2:
      CUDA_SAFE_CALL(cuMemsetD16(dst, m_paddingValue, numberElementsMemset);
      break;
    case 4:
      CUDA_SAFE_CALL(cuMemsetD32(dst, m_paddingValue, numberElementsMemset);
      break;
    default:
      throw invalid_argument("invalid elementsize of paddingArg. Only 1,2 or 4 bytes elementsize are supported.");
  }
}

void DataCopyKernelArgMatrixPadding::copyDataDtoH() {
  CUdeviceptr dst = m_kernelArg->m_ddata;
  char *src = static_cast<char *>(const_cast<void *>(m_kernelArg->m_hdata));

  for (int i = 0; i < m_kernelArg->m_dst_rows; i++) {
    CUDA_SAFE_CALL(cuMemcpyDtoH(src, dst, m_kernelArg->m_dst_columns));
    dst += m_kernelArg->m_dst_columns;
    src += m_kernelArg->m_src_columns;
  }
}