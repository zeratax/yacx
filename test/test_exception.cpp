#include "yacx/Exception.hpp"
#include <catch2/catch.hpp>

TEST_CASE("NVRTC_SAFE_CALL", "[yacx::Exception]") {

  SECTION("NVRTC_SUCCESS") {
    try {
      yacx::NVRTC_SAFE_CALL(NVRTC_SUCCESS);
    } catch (yacx::nvrtcResultException &e) {
      REQUIRE(e.type == NVRTC_SUCCESS);
      FAIL("Correct nvrtcResultException, but it shouldn't be thrown an "
           "exception");
    } catch (std::exception &e) {
      FAIL("Exception != nvrtcResultException, but it shouldn't be thrown an "
           "exception");
    }
  }

  SECTION("NVRTC_ERROR_OUT_OF_MEMORY") {
    try {
      yacx::NVRTC_SAFE_CALL(NVRTC_ERROR_OUT_OF_MEMORY);
    } catch (yacx::nvrtcResultException &e) {
      REQUIRE(e.type == NVRTC_ERROR_OUT_OF_MEMORY);
    } catch (std::exception &e) {
      FAIL("Exception != nvrtcResultException");
    }
  }

  SECTION("NVRTC_ERROR_INVALID_INPUT") {
    try {
      yacx::NVRTC_SAFE_CALL(nvrtcVersion(nullptr, nullptr));
    } catch (yacx::nvrtcResultException &e) {
      REQUIRE(e.type == NVRTC_ERROR_INVALID_INPUT);
    } catch (std::exception &e) {
      FAIL("Exception != nvrtcResultException");
    }
  }
}

TEST_CASE("CUDA_SAFE_CALL", "[yacx::Exception]") {

  SECTION("CUDA_SUCCESS") {
    try {
      yacx::CUDA_SAFE_CALL(CUDA_SUCCESS);
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_SUCCESS);
      FAIL(
          "Correct CUresultException, but it shouldn't be thrown an exception");
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException, but it shouldn't be thrown an "
           "exception");
    }
  }

  SECTION("CUDA_ERROR_PROFILER_DISABLED") {
    try {
      yacx::CUDA_SAFE_CALL(CUDA_ERROR_PROFILER_DISABLED);
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_PROFILER_DISABLED);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }

  SECTION("CUDA_ERROR_INVALID_PC") {
    try {
      yacx::CUDA_SAFE_CALL(CUDA_ERROR_INVALID_PC);
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_INVALID_PC);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }

  SECTION("CUDA_ERROR_INVALID_VALUE") {
    try {
      yacx::CUDA_SAFE_CALL(cuCtxDestroy(nullptr));
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_INVALID_VALUE);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }
}
