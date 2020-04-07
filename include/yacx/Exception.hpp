#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h> // z.B. für dim3

#include "yacx/Logger.hpp" //um den logger benutzen zu können

namespace yacx {

namespace detail {

//!
//! \return  the number of columns of the terminal
unsigned int askTerminalSize();

//! split a string in C++
//! source: <a
//! link="http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html">www.martinbroadhurst.com</a>
//! \tparam Container
//! \param str
//! \param cont
//! \param delim
template <class Container>
void split(const std::string &str, Container &cont, char delim = ' ');

//! this function indents the lines of the descripton to make look nicer
//! \param desc
//! \return
std::string descriptionFkt(const std::string &desc);

//!
//! \param error see <a
//! href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9">CUDA
//! Driver API Documentation</a> \return description of error
std::string whichError(const CUresult &error);

//! more info see <a
//! href="https://github.com/ptillet/isaac/blob/master/include/isaac/external/CUDA/nvrtc.h">NVRTC
//! Github Repository</a> \param error see <a
//! href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1g31e41ef222c0ea75b4c48f715b3cd9f0">NVRTC
//! Documentation</a> \return description of error
std::string whichError(const nvrtcResult &error);

} // namespace detail

/*!
  \class nvidiaException Exception.hpp
  \brief superclass of nvrtcResultException and cudaResultException
  \tparam error descripton of error
  \example docs/cudaexeption.cpp
*/
class nvidiaException : public std::exception {
 protected:
  std::string error;

 public:
  const char *what() const throw() { return error.c_str(); }
};

/*!
  \class nvrtcResultException Exception.hpp
  \brief Exception class for NVRTC errors
  \tparam type Name and type of the NVRTC Error, e.g.
  <code>NVRTC_ERROR_OUT_OF_MEMORY</code>
  \tparam error description of the NVRTC Error
  \example docs/nvrtcexception.cpp
*/
class nvrtcResultException : public nvidiaException {
 public:
  nvrtcResult type;
  nvrtcResultException(nvrtcResult type, std::string error) {
    this->type = type;
    this->error = error;
    logger(loglevel::WARNING)
        << "nvrtcResultException " << (int)type
        << " with description: " << this->error.c_str() << " created.";
  }
};

/*!
  \class CUresultException Exception.hpp
  \brief Exception class for CUDA driver api errors
  \tparam type Name and type of the CUDA Error, e.g.
  <code>CUDA_ERROR_NO_DEVICE</code> \tparam error descripton of error \example
  docs/cudaexeption.cpp
*/
class CUresultException : public nvidiaException {
 public:
  CUresult type;
  CUresultException(CUresult type, std::string error) {
    this->type = type;
    this->error = error;
    logger(loglevel::WARNING)
        << "CUresultException " << (int)type
        << " with description: " << this->error.c_str() << " created.";
  }
};

//! throws a nvrtcResultException if something went wrong
#define NVRTC_SAFE_CALL(error)                                                 \
  __checkNvrtcResultError(error, __FILE__, __LINE__);
inline void __checkNvrtcResultError(const nvrtcResult error, const char *file,
                                    const int line) {
  if (NVRTC_SUCCESS != error) {
    // create string for exception
    std::string exception =
        nvrtcGetErrorString(error); // method to get the error name from NVIDIA
    exception = exception + "\n->occoured in file <" + file + " in line " +
                std::to_string(line) + "\n\n";
    throw nvrtcResultException(error, exception);
  }
}

//! throws a nvrtcResultException if something went wrong
#define NVRTC_SAFE_CALL_LOG(error, m_log)                                      \
  __checkNvrtcResultError_LOG(error, m_log, __FILE__, __LINE__);
inline void __checkNvrtcResultError_LOG(const nvrtcResult error,
                                        std::__cxx11::basic_string<char> m_log,
                                        const char *file, const int line) {
  if (NVRTC_SUCCESS != error) {
    // create string for exception
    std::string exception =
        nvrtcGetErrorString(error); // method to get the error name from NVIDIA
    exception = exception + "\n->occoured in file <" + file + " in line " +
                std::to_string(line) + "\n m_log: " + m_log + "\n\n";
    throw nvrtcResultException(error, exception);
  }
}

//! throws a CUresultException if something went wrong
#define CUDA_SAFE_CALL(error)                                                  \
  yacx::__checkCUresultError(error, __FILE__, __LINE__);
inline void __checkCUresultError(const CUresult error, const char *file,
                                 const int line) {
  if (CUDA_SUCCESS != error) {
    // create string for exception
    const char *name;
    cuGetErrorName(error, &name); // method to get the error name from NVIDIA
    const char *description;
    cuGetErrorString(
        error, &description); // method to get the error description from NVIDIA
    std::string exception = name;
    exception = exception + "\n->occoured in file <" + file + " in line " +
                std::to_string(line) + "\n" + "  ->" + description +
                "\n"
                "  ->" +
                detail::whichError(error).c_str() + "\n\n";
    // choose which error to throw
    throw CUresultException(error, exception);
  }
}

} // namespace yacx
