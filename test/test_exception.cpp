#include "yacx/Exception.hpp"
#include <catch2/catch.hpp>
#include <string>

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
      CUDA_SAFE_CALL(CUDA_SUCCESS);
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
      CUDA_SAFE_CALL(CUDA_ERROR_PROFILER_DISABLED);
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_PROFILER_DISABLED);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }

  SECTION("CUDA_ERROR_INVALID_PC") {
    try {
      CUDA_SAFE_CALL(CUDA_ERROR_INVALID_PC);
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_INVALID_PC);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }

  SECTION("CUDA_ERROR_INVALID_VALUE") {
    try {
      CUDA_SAFE_CALL(cuCtxDestroy(nullptr));
    } catch (yacx::CUresultException &e) {
      REQUIRE(e.type == CUDA_ERROR_INVALID_VALUE);
    } catch (std::exception &e) {
      FAIL("Exception != CUresultException");
    }
  }
}

//_____________________FELIX_CODE_________________
// SCENARIO("Exceptions are tested under the following conditions.") {
TEST_CASE("1. Selected nvrtc-Exceptions are returned.") {
  SECTION("1A. nvrtc-Success:") {
    const nvrtcResult error = NVRTC_SUCCESS;
    std::string nvrtc_SUCCESS_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_SUCCESS_string == "NO_Error: 0~NVRTC_SUCCESS");
  }
  SECTION("1B. NVRTC_ERROR_OUT_OF_MEMORY:") {
    const nvrtcResult error = NVRTC_ERROR_OUT_OF_MEMORY;
    std::string nvrtc_outmemory_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_outmemory_string == "1~NVRTC_ERROR_OUT_OF_MEMORY");
  }

  SECTION("1C. NVRTC_ERROR_PROGRAM_CREATION_FAILURE:") {
    const nvrtcResult error = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
    std::string nvrtc_creation_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_creation_string == "2~NVRTC_ERROR_PROGRAM_CREATION_FAILURE");
  }
  SECTION("1D. NVRTC_ERROR_INVALID_INPUT:") {
    const nvrtcResult error = NVRTC_ERROR_INVALID_INPUT;
    std::string nvrtc_input_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_input_string == "3~NVRTC_ERROR_INVALID_INPUT");
  }
  SECTION("1E. NVRTC_ERROR_INVALID_PROGRAM:") {
    const nvrtcResult error = NVRTC_ERROR_INVALID_PROGRAM;
    std::string nvrtc_program_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_program_string == "4~NVRTC_ERROR_INVALID_PROGRAM");
  }
  SECTION("1F. NVRTC_ERROR_INVALID_OPTION:") {
    const nvrtcResult error = NVRTC_ERROR_INVALID_OPTION;
    std::string nvrtc_ioptions_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_ioptions_string == "5~NVRTC_ERROR_INVALID_OPTION");
  }

  SECTION("1G. NVRTC_ERROR_COMPILATION:") {
    const nvrtcResult error = NVRTC_ERROR_COMPILATION;
    std::string nvrtc_compilation_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_compilation_string == "6~NVRTC_ERROR_COMPILATION");
  }

  SECTION("1H. NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:") {
    const nvrtcResult error = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
    std::string nvrtc_boperation_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_boperation_string ==
            "7~NVRTC_ERROR_BUILTIN_OPERATION_FAILURE");
  }

  SECTION("1J. NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:") {
    const nvrtcResult error = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
    std::string nvrtc_acompilation_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_acompilation_string ==
            "8~NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION");
  }

  SECTION("1K. NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:") {
    const nvrtcResult error = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
    std::string nvrtc_lbcompilation_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_lbcompilation_string ==
            "9~NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION");
  }

  SECTION("1L. NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:") {
    const nvrtcResult error = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
    std::string nvrtc_expression_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_expression_string ==
            "10~NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID");
  }

  SECTION("1M. NVRTC_ERROR_INTERNAL_ERROR:") {
    const nvrtcResult error = NVRTC_ERROR_INTERNAL_ERROR;
    std::string nvrtc_internal_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_internal_string == "11~NVRTC_ERROR_INTERNAL_ERROR");
  }
  SECTION("1N. Others:") {
    const nvrtcResult error = (const nvrtcResult)41;
    std::string nvrtc_other_string = yacx::detail::whichError(error);
    REQUIRE(nvrtc_other_string == "Error: 41~error_unknown");
  }
}
//}
//_____________________FELIX_CODE_________________
