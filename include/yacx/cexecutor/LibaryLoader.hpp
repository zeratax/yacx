#pragma once

#include <string>

namespace yacx {
namespace detail {
struct dynop {
  void (*op)(void **parameter);
  void *libhandle;
};

struct opfn {
  void (*op)(void **parameter);
};

//! loads a libary and search a specific operation
//! \param dest struct to store information
//! \param filename name of the libary
//! \param opSymbolName name of the opfn-struct containg the operation, wich should be loaded
void load_op(struct dynop *dest, std::string filename, std::string opSymbolName);
//! unloads the libary and cleans the struct
//! \param op 
void unload_op(struct dynop *op);
} // namespace detail
} // namespace yacx