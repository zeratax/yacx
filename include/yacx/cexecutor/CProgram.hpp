#pragma once

#include <fstream>

namespace yacx {
class CProgram {
 public:
  CProgram(char *cProgram, char *functionName, int numberParameters,
           char *compilerWithOptions);
  ~CProgram();

  void execute(void **arguments);

 private:
  void createSrcFile(char *cProgram, char *functionName, int numberParameters,
                     std::ofstream &fileOut);
  void compile(char *cProgram, char *functionName, int numberParameters,
               char *compilerWithOptions = "gcc -Wall");

  static int id;
  struct dynop op;
  const char *srcFile;
  const char *libFile;
};
} // namespace yacx