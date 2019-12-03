#include "cudaexecutor/KernelArg.hpp"
#include "cudaexecutor/Exception.hpp"
#include "cudaexecutor/Logger.hpp"

using cudaexecutor::KernelArg, cudaexecutor::loglevel;

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
      CUDA_SAFE_CALL(cuEventCreate(&end, CU_EVENT_DEFAULT));

      CUDA_SAFE_CALL(cuEventRecord(start, 0));
      CUDA_SAFE_CALL(
          cuMemcpyHtoD(m_ddata, const_cast<void *>(m_hdata), m_size));
      CUDA_SAFE_CALL(cuEventRecord(end, 0));

      CUDA_SAFE_CALL(cuEventSynchronize(stop));
      CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, end));
    }
  } else {
    logger(loglevel::DEBUG1) << "NOT uploading argument";
  }

  return time;
}

float KernelArg::download() {
  float time{0};

  if (m_download) {
    logger(loglevel::DEBUG1) << "downloading argument";

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_SAFE_CALL(cuEventCreate(&end, CU_EVENT_DEFAULT));

    CUDA_SAFE_CALL(cuEventRecord(start, 0));
    CUDA_SAFE_CALL(cuMemcpyDtoH(const_cast<void *>(m_hdata), m_ddata, m_size));
    CUDA_SAFE_CALL(cuEventRecord(end, 0));

    CUDA_SAFE_CALL(cuEventSynchronize(stop));
    CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, end));
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

const void *KernelArg::content() {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "returning device pointer";
    return &m_ddata;
  }
  logger(loglevel::DEBUG1) << "returning host pointer";
  return m_hdata;
}