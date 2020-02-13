#pragma once

namespace yacx {
namespace detail {
struct dynop {
  void (*op)(void **parameter);
  void *libhandle;
};

struct opfn {
  void (*op)(void **parameter);
};

void load_op(struct dynop *dest, const char *filename);
void unload_op(struct dynop *op);
} // namespace detail
} // namespace yacx