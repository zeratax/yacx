#if _MSC_VER
#define popen _popen
#define pclose _pclose
#endif

#include "yacx/Devices.hpp"
#include "yacx/Exception.hpp"

#include <array>
#include <catch2/catch.hpp>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using yacx::Device, yacx::Devices;

// https://stackoverflow.com/a/478960
std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

TEST_CASE("Device can be constructed", "[yacx::device]") {
  try {
    #if defined(_MSC_VER)
    std::string name = exec("powershell -command \"wmic path win32_VideoController get name | "
                            "Select-String -Pattern 'NVIDIA ([a-zA-Z0-9\-]+)' "
                            "-AllMatches | % { $_.matches.groups[1].value }\"");
    #else
    std::string name =
        exec("lspci | grep -Poi \"nvidia.+\\[\\K[a-zA-Z0-9 ]+(?=\\])\"");
    #endif
    name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());

    SECTION("first device") {
      Device dev = Devices::findDevice();
      REQUIRE(dev.name() == name);
    }
    SECTION("by name") {
      Device dev = Devices::findDevice(name);
      REQUIRE(dev.name() == name);
      REQUIRE_THROWS_AS(
          [&]() {
            Device dev = Devices::findDevice(std::string{"Radeon RX Vega 64"});
          }(),
          std::invalid_argument);
    }

  } catch (yacx::CUresultException &e) {
    switch (e.type) {
    case CUDA_ERROR_NO_DEVICE:
      // e.what();
      FAIL("you probably don't have a CUDA-capable device, or the CUDA-driver "
           "couldn't detect it");
      break;
    default:
      e.what();
      break;
    }
  }
}
