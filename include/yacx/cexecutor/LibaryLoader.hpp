#pragma once

namespace yacx {
namespace detail {
struct dynop {
  void (*op)(void **parameter, bool* pointerArg);
  void *libhandle;
};

struct opfn {
  void (*op)(void **parameter, bool* pointerArg);
};

//! loads a libary and search a specific operation
//! \param dest struct to store information
//! \param filename name of the libary
void load_op(struct dynop *dest, const char *filename);
//! unloads the libary and cleans the struct
//! \param op 
void unload_op(struct dynop *op);
} // namespace detail
} // namespace yacx