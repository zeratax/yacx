#pragma once

#include "../../include/yacx/JNIHandle.hpp"
#include "../../include/yacx/KernelArgs.hpp"

#include <memory>
#include <string>

namespace jni {
namespace detail {
//! Wrapper class to wrap pointer to pagelocked memory allocated with CUDA
//! Driver API
class HDataMem {
 public:
  //! Allocate page-locked memory with passed size
  HDataMem(size_t size);
  //! Frees allocated page-locked memory
  ~HDataMem();

  //! \return returns pointer to page-locked memory
  void *get() { return m_hdata; };

 private:
  void *m_hdata;
};
} // namespace detail

class KernelArgJNI : yacx::JNIHandle {
  friend class KernelArgJNISlice;

 public:
  KernelArgJNI(std::shared_ptr<detail::HDataMem> hdata,
               yacx::KernelArg *kernelArg, std::string type)
      : m_hdata(hdata), m_kernelArg(kernelArg), m_type(type){};
  KernelArgJNI(size_t size, bool download, bool copy, bool upload,
               std::string type);
  ~KernelArgJNI();
  yacx::KernelArg *kernelArgPtr() { return m_kernelArg; }
  virtual void *getHostData() const { return m_hdata.get()->get(); }
  std::shared_ptr<detail::HDataMem> getHostDataSharedPointer() {
    return m_hdata;
  }
  std::string &getType() { return m_type; }

 private:
  std::shared_ptr<detail::HDataMem> m_hdata;
  yacx::KernelArg *m_kernelArg;
  std::string m_type;
};

class KernelArgJNISlice : public KernelArgJNI {
 public:
  KernelArgJNISlice(size_t start, size_t end, KernelArgJNI *arg);
  void *getHostData() const override {
    return reinterpret_cast<char *>(m_hdata.get()->get()) + m_offset;
  }

 private:
  size_t m_offset;
};
} // namespace jni