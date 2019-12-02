#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h> // z.B. für dim3

namespace cudaexecutor {

namespace detail {

//!
//! \return  the number of columns of the terminal
unsigned int askTerminalSize();

//! split a string in C++
//! source: <a link="http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html">www.martinbroadhurst.com</a>
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
//! \param error see <a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9">CUDA Driver API Documentation</a>
//! \return description of error
std::string whichError(const CUresult &error);

//! more info see <a href="https://github.com/ptillet/isaac/blob/master/include/isaac/external/CUDA/nvrtc.h">NVRTC Github Repository</a>
//! \param error see <a href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1g31e41ef222c0ea75b4c48f715b3cd9f0">NVRTC Documentation</a>
//! \return description of error
std::string whichError(const nvrtcResult &error);

} // namespace detail

/*!
  \class CUresultException Exception.hpp
  \brief Exception class for CUDA driver api errors
  \tparam name Name of the CUDA Error, e.g. <code>CUDA_ERROR_NO_DEVICE</code>
  \example cudaexception.cpp
*/
template <CUresult name = (CUresult)0>
class CUresultException : public std::exception {
 public:
  //!
  //! \param error description of exception
  explicit CUresultException(const std::string &error) { this->error = error; }
  //!
  //! \return description of exception
  [[nodiscard]] const char *what() const noexcept override {
    return error.c_str();
  }

 private:
  std::string error;
};

/*!
  \class nvrtcResultException Exception.hpp
  \brief Exception class for NVRTC errors
  \tparam name Name of the NVRTC Error, e.g.
  <code>NVRTC_ERROR_OUT_OF_MEMORY</code>
  \example nvrtcexception.cpp
*/
template <nvrtcResult name = (nvrtcResult)0>
class nvrtcResultException : public std::exception {
 private:
  std::string error;

 public:
  explicit nvrtcResultException(const std::string &error) {
    this->error = error;
  }
  [[nodiscard]] const char *what() const noexcept override {
    return error.c_str();
  }
};

//! throws a nvrtcResultException if something went wrong
#define NVRTC_SAFE_CALL(error)                                                 \
  __checkNvrtcResultError(error, __FILE__, __LINE__);
inline void __checkNvrtcResultError(const nvrtcResult error, const char *file,
                                    const int line) {
  if (NVRTC_SUCCESS != error) {
    // create string for exception
    std::string exception = detail::whichError(error);
    exception = exception + "\n->occoured in file:[" + file + ":" +
                std::to_string(line) + "]\n";
    // choose which error to throw
    switch (error) {
    case 0:
      printf("no error caught in line %i in file %s\n", line, file);
    case 1: // NVRTC_ERROR_OUT_OF_MEMORY
      throw nvrtcResultException<NVRTC_ERROR_OUT_OF_MEMORY>(exception);
    case 2: // NVRTC_ERROR_PROGRAM_CREATION_FAILURE
      throw nvrtcResultException<NVRTC_ERROR_PROGRAM_CREATION_FAILURE>(
          exception);
    case 3: // NVRTC_ERROR_INVALID_INPUT
      throw nvrtcResultException<NVRTC_ERROR_INVALID_INPUT>(exception);
    case 4: // NVRTC_ERROR_INVALID_PROGRAM
      throw nvrtcResultException<NVRTC_ERROR_INVALID_PROGRAM>(exception);
    case 5: // NVRTC_ERROR_INVALID_OPTION
      throw nvrtcResultException<NVRTC_ERROR_INVALID_OPTION>(exception);
    case 6: // NVRTC_ERROR_COMPILATION
      throw nvrtcResultException<NVRTC_ERROR_COMPILATION>(exception);
    case 7: // NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
      throw nvrtcResultException<NVRTC_ERROR_BUILTIN_OPERATION_FAILURE>(
          exception);
    case 8: // NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
      throw nvrtcResultException<
          NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION>(exception);
    case 9: // NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
      throw nvrtcResultException<
          NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION>(exception);
    case 10: // NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
      throw nvrtcResultException<NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID>(
          exception);
    case 11: // NVRTC_ERROR_INTERNAL_ERROR
      throw nvrtcResultException<NVRTC_ERROR_INTERNAL_ERROR>(exception);
    default:
      printf("CUresult Error is unknown\n");
      exit(-1);
    }
  }
}

//! throws a CUresultException if something went wrong
#define CUDA_SAFE_CALL(error) __checkCUresultError(error, __FILE__, __LINE__);
inline void __checkCUresultError(const CUresult error, const char *file,
                                 const int line) {
  if (CUDA_SUCCESS != error) {
    // create string for exception
    std::string exception = detail::whichError(error);
    exception = exception + "\n->occoured in file:[" + file + ":" +
                std::to_string(line) + "]\n";
    // choose which error to throw
    switch (error) {
    case 0:
      printf("no error caught in line %i in file %s\n", line, file);
    case 1:
      throw CUresultException<(CUresult)1>(exception);
    case 2:
      throw CUresultException<(CUresult)2>(exception);
    case 3:
      throw CUresultException<(CUresult)3>(exception);
    case 4:
      throw CUresultException<(CUresult)4>(exception);
    case 5:
      throw CUresultException<(CUresult)5>(exception);
    case 6:
      throw CUresultException<(CUresult)6>(exception);
    case 7:
      throw CUresultException<(CUresult)7>(exception);
    case 8:
      throw CUresultException<(CUresult)8>(exception);
    case 100:
      throw CUresultException<(CUresult)100>(exception);
    case 101:
      throw CUresultException<(CUresult)101>(exception);
    case 200:
      throw CUresultException<(CUresult)200>(exception);
    case 201:
      throw CUresultException<(CUresult)201>(exception);
    case 202:
      throw CUresultException<(CUresult)202>(exception);
    case 205:
      throw CUresultException<(CUresult)205>(exception);
    case 206:
      throw CUresultException<(CUresult)206>(exception);
    case 207:
      throw CUresultException<(CUresult)207>(exception);
    case 208:
      throw CUresultException<(CUresult)208>(exception);
    case 209:
      throw CUresultException<(CUresult)209>(exception);
    case 210:
      throw CUresultException<(CUresult)210>(exception);
    case 211:
      throw CUresultException<(CUresult)211>(exception);
    case 212:
      throw CUresultException<(CUresult)212>(exception);
    case 213:
      throw CUresultException<(CUresult)213>(exception);
    case 214:
      throw CUresultException<(CUresult)214>(exception);
    case 215:
      throw CUresultException<(CUresult)215>(exception);
    case 216:
      throw CUresultException<(CUresult)216>(exception);
    case 217:
      throw CUresultException<(CUresult)217>(exception);
    case 218:
      throw CUresultException<(CUresult)218>(exception);
    case 219:
      throw CUresultException<(CUresult)219>(exception);
    case 220:
      throw CUresultException<(CUresult)220>(exception);
    case 221:
      throw CUresultException<(CUresult)221>(exception);
    case 300:
      throw CUresultException<(CUresult)300>(exception);
    case 301:
      throw CUresultException<(CUresult)301>(exception);
    case 302:
      throw CUresultException<(CUresult)302>(exception);
    case 303:
      throw CUresultException<(CUresult)303>(exception);
    case 304:
      throw CUresultException<(CUresult)304>(exception);
    case 400:
      throw CUresultException<(CUresult)400>(exception);
    case 500:
      throw CUresultException<(CUresult)500>(exception);
    case 600:
      throw CUresultException<(CUresult)600>(exception);
    case 700:
      throw CUresultException<(CUresult)700>(exception);
    case 701:
      throw CUresultException<(CUresult)701>(exception);
    case 702:
      throw CUresultException<(CUresult)702>(exception);
    case 703:
      throw CUresultException<(CUresult)703>(exception);
    case 704:
      throw CUresultException<(CUresult)704>(exception);
    case 705:
      throw CUresultException<(CUresult)705>(exception);
    case 708:
      throw CUresultException<(CUresult)708>(exception);
    case 709:
      throw CUresultException<(CUresult)709>(exception);
    case 710:
      throw CUresultException<(CUresult)710>(exception);
    case 711:
      throw CUresultException<(CUresult)711>(exception);
    case 712:
      throw CUresultException<(CUresult)712>(exception);
    case 713:
      throw CUresultException<(CUresult)713>(exception);
    case 714:
      throw CUresultException<(CUresult)714>(exception);
    case 715:
      throw CUresultException<(CUresult)715>(exception);
    case 716:
      throw CUresultException<(CUresult)716>(exception);
    case 717:
      throw CUresultException<(CUresult)717>(exception);
    case 718:
      throw CUresultException<(CUresult)718>(exception);
    case 719:
      throw CUresultException<(CUresult)719>(exception);
    case 720:
      throw CUresultException<(CUresult)720>(exception);
    case 800:
      throw CUresultException<(CUresult)800>(exception);
    case 801:
      throw CUresultException<(CUresult)801>(exception);
    case 999:
      throw CUresultException<(CUresult)999>(exception);
    default:
      printf("CUresult Error is unknown\n");
      exit(-1);
    }
  }
}

} // namespace cudaexecutor
