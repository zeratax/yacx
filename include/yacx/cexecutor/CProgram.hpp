#pragma once

#include "../JNIHandle.hpp"
#include "LibaryLoader.hpp"
#include <fstream>

namespace yacx {
class CProgram : public JNIHandle {
 public:
  CProgram(const char *cProgram, const char *functionName, int numberParameters,
           const char *compilerWithOptions);
  ~CProgram();

  void execute(void **arguments);
  int getNumberArguments() const { return m_numberArguments; }

 private:
  void createSrcFile(const char *cProgram, const char *functionName,
                     int numberParameters, std::ofstream &fileOut);
  void compile(const char *cProgram, const char *functionName,
               int numberParameters,
               const char *compilerWithOptions = "gcc -Wall");

  static int id;
  int m_numberArguments;
  struct detail::dynop m_op;
  const char *m_srcFile;
  const char *m_libFile;
};
} // namespace yacx