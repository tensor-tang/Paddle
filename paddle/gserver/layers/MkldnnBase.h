/* Copyright (c) 2016 */

#pragma once

#include "mkldnn.hpp"

namespace paddle {

static const std::string DNN_FMTS[] = {
  "undef", "any", "blocked", "x", "nc", "nchw", "nhwc", "chwn", "nChw8c",
  "nChw16c", "oi", "io", "oihw", "ihwo", "OIhw8i8o", "OIhw16i16o", "OIhw8o8i",
  "OIhw16o16i", "Ohwi8o", "Ohwi16o", "goihw", "gOIhw8i8o", "gOIhw16i16o",
  "gOIhw8o8i", "gOIhw16o16i"};
  // mkldnn_oIhw8i = mkldnn_nChw8c
  // mkldnn_oIhw16i = mkldnn_nChw16c

/// For dnn engine
class CpuEngine {
public:
  static CpuEngine & Instance() {
    // I's thread-safe in C++11.
    static CpuEngine myInstance;
    return myInstance;
  }
  CpuEngine(CpuEngine const&) = delete;             // Copy construct
  CpuEngine(CpuEngine&&) = delete;                  // Move construct
  CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
  CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

  mkldnn::engine & getEngine() { return cpuEngine_; }
protected:
  CpuEngine() : cpuEngine_(mkldnn::engine::cpu, 0) {}
//    CpuEngine() : _cpu_engine(engine::cpu_lazy, 0) {}
  ~CpuEngine() {}
private:
  mkldnn::engine cpuEngine_;
};

}  // namespace paddle
