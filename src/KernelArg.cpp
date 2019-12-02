#include "cudaexecutor/KernelArg.hpp"
#include "cudaexecutor/Exception.hpp"
#include "cudaexecutor/Logger.hpp"

using cudaexecutor::KernelArg, cudaexecutor::loglevel;

KernelArg::KernelArg(void *const data, size_t size, bool download, bool copy,
                       bool upload)
    : m_hdata{data}, m_size{size},
      m_download{download}, m_copy{copy}, m_upload{
                                                                       upload} {
  logger(loglevel::DEBUG) << "created KernelArg with size: " << size
                          << ", which should " << (m_upload ? "be" : "not be")
                          << " uploaded and should "
                          << (m_download ? "be" : "not be") << " downloaded";
}

void KernelArg::upload() {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "uploading argument";
    CUDA_SAFE_CALL(cuMemAlloc(&m_ddata, m_size));
    if (m_copy) {
      logger(loglevel::DEBUG1) << "copying data to device";
      CUDA_SAFE_CALL(cuMemcpyHtoD(m_ddata, const_cast<void *>(m_hdata), m_size));
    }
  } else {
    logger(loglevel::DEBUG1) << "NOT uploading argument";
  }
}

void KernelArg::download() {
  if (m_download) {
    logger(loglevel::DEBUG1) << "downloading argument";
    CUDA_SAFE_CALL(cuMemcpyDtoH(const_cast<void *>(m_hdata), m_ddata, m_size));
  } else {
    logger(loglevel::DEBUG1) << "NOT downloading argument";
  }

  if (m_upload) {
    logger(loglevel::DEBUG1) << "freeing argument from device";
    CUDA_SAFE_CALL(cuMemFree(m_ddata));
  } else {
    logger(loglevel::DEBUG1) << "NOT freeing argument from device";
  }
}

const void *KernelArg::content() {
  if (m_upload) {
    logger(loglevel::DEBUG1) << "returning device pointer";
    return &m_ddata;
  }
  logger(loglevel::DEBUG1) << "returning host pointer";
  return m_hdata;
}