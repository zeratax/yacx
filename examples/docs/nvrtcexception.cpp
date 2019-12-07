#include "yacx/Exception.hpp"
#include <nvrtc.h>

using yacx::nvrtcResultException;

try {
  NVRTC_SAFE_CALL(nvrtcCompileProgram(nullptr, 0, NULL));
} catch (nvrtcResultException<(nvrtcResult)1> &e) {
  std::cout << "Wrong Exception caught" << std::endl;
  std::cout << e.what() << std::endl;
} catch (nvrtcResultException<NVRTC_ERROR_COMPILATION> &e) {
  std::cout << "Correct Exception caught" << std::endl;
  std::cout << e.what() << std::endl;
} catch (std::exception &e) {
  // Other errors
  std::cout << "other Error\n";
  std::cout << e.what() << std::endl;
}