#pragma once

#include "../JNIHandle.hpp"
#include "../Options.hpp"
#include "LibaryLoader.hpp"
#include <fstream>
#include <vector>

namespace yacx {
class CProgram : public JNIHandle {
 public:
  //! Constructs new compiled CProgram. <br>
  //! The decleration of structs, functions or variables with one of the
  //! following name within passed C-Code is permitted: <br> op<functionName>,
  //! opfn<functionName>, execute<functionName> \param cProgram C-Code for
  //! program \param functionName name of c-function, which should be executed
  //! \param parameterTypes type of the parameters e.g <code>int</code> or
  //! <code>float*</code> <br> pointer types can be abbreviated by * \param
  //! compiler name of the compiler, which should be used to compile this
  //! cProgram \param options options for the compiler
  CProgram(const char *cProgram, const char *functionName,
           std::vector<std::string> &parameterTypes,
           const char *compiler = "gcc", Options &options = DEFAULT_OPTIONS);
  ~CProgram();

  //! Executes a the cFunction
  //! \param arguments arguments for the cFunction
  void execute(std::vector<void *> arguments);

 private:
  void createSrcFile(const char *cProgram, const char *functionName,
                     std::vector<std::string> &parameterTypes,
                     std::ofstream &fileOut);
  void compile(const char *cProgram, const char *functionName,
               std::vector<std::string> &parameterTypes,
               std::string &compilerCommand);

  static int id;
  static Options DEFAULT_OPTIONS;
  unsigned int m_numberArguments;
  struct detail::dynop m_op;
  std::string m_srcFile;
  std::string m_libFile;
};
} // namespace yacx