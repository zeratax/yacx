#include "../include/catch2/catch.hpp"

#include "../include/cudaexecutor/Device.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

using cudaexecutor::Device;


std::string exec(const char* cmd) {
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

TEST_CASE("Device can be constructed", "[cudaexecutor::device]") {
  Device dev;

  REQUIRE(dev.name() == exec("lspci | grep -Eoi \"(nvidia.+)\""));
}
