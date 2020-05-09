// Copyright 2019 André Hodapp
#include "yacx/Exception.hpp"
#if _MSC_VER
#define popen _popen
#define pclose _pclose
#endif

/*
 * returns the number of columns of the terminal
 */
unsigned int yacx::detail::askTerminalSize() {
  // use a file fp to save and get output of system command
  FILE *fp;
  char var[40];

  //"tput cols" tells you the number of columns

  fp = popen("tput cols", "r");
  while (fgets(var, sizeof(var), fp) != nullptr) {
    // printf("command line has columns size: %s\n", var);
  }
  pclose(fp);
  unsigned int ret = atoi(var);
  // system("tput cols");
  return ret;
}

/*
 * split a string in C++
 * source: http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html
 */
template <class Container>
void yacx::detail::split(const std::string &str, Container &cont, char delim) {
  std::size_t current, previous = 0;
  current = str.find(delim);
  while (current != std::string::npos) {
    cont.push_back(str.substr(previous, current - previous));
    previous = current + 1;
    current = str.find(delim, previous);
  }
  cont.push_back(str.substr(previous, current - previous));
}

/*
 * this function indents the lines of the descripton to make look nicer :)
 */
std::string yacx::detail::descriptionFkt(const std::string &desc) {
  // size of each line is defined by the terminal size -24
  const unsigned int size_of_line = askTerminalSize() - 24;

  // split string in words
  std::vector<std::string> substrings;
  split(desc, substrings);

  // variables
  char indent[] = "        ";
  std::string ret = "-->description: \n";
  ret = ret + indent;
  unsigned int size_of_currentLine = 0;

  // Am Anfang steht Deprecated
  if (substrings.at(0).compare("Deprecated") == 0) {
    ret = ret + " " + "|_DEPRECATED_|" + "\n";
  } else {
    ret = ret + " " + substrings.at(0);
    size_of_currentLine += substrings.at(0).size();
  }

  // put words into a string which has lines with
  // length of size_of_line
  for (long unsigned int i = 1; i < substrings.size(); ++i) {
    if (size_of_currentLine + substrings.at(i).size() > size_of_line) {
      ret = ret + "\n" + indent;
      size_of_currentLine = 0;
    }
    ret = ret + " " + substrings.at(i);
    size_of_currentLine += substrings.at(i).size();
  }
  return ret;
}

/*
 * returns the type of nvrtc error
 * the different errors are in :
 * https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1g31e41ef222c0ea75b4c48f715b3cd9f0
 * and the program itself is in
 * https://github.com/ptillet/isaac/blob/master/include/isaac/external/CUDA/nvrtc.h
 */
// Ignore -Wswitch
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
std::string yacx::detail::whichError(const nvrtcResult &error) {
  std::string ret{"Error: "};
  ret.append(std::to_string(error));
  ret.append("~");
  std::string description;
  switch ((unsigned int)error) {
  case 0:
    ret = "NO_Error: 0~NVRTC_SUCCESS";
    return ret;
  case 1: // NVRTC_ERROR_OUT_OF_MEMORY
    ret = "1~NVRTC_ERROR_OUT_OF_MEMORY";
    return ret;
  case 2: // NVRTC_ERROR_PROGRAM_CREATION_FAILURE
    ret = "2~NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    return ret;
  case 3: // NVRTC_ERROR_INVALID_INPUT
    ret = "3~NVRTC_ERROR_INVALID_INPUT";
    return ret;
  case 4: // NVRTC_ERROR_INVALID_PROGRAM
    ret = "4~NVRTC_ERROR_INVALID_PROGRAM";
    return ret;
  case 5: // NVRTC_ERROR_INVALID_OPTION
    ret = "5~NVRTC_ERROR_INVALID_OPTION";
    return ret;
  case 6: // NVRTC_ERROR_COMPILATION
    ret = "6~NVRTC_ERROR_COMPILATION";
    return ret;
  case 7: // NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
    ret = "7~NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    return ret;
  case 8: // NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    ret = "8~NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    return ret;
  case 9: // NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    ret = "9~NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    return ret;
  case 10: // NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    ret = "10~NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    return ret;
  case 11: // NVRTC_ERROR_INTERNAL_ERROR
    ret = "11~NVRTC_ERROR_INTERNAL_ERROR";
    return ret;
  default:
    ret.append("error_unknown");
    description = "''It's a trap! Abort! We never left the CPU o(╥﹏╥)o'' But "
                  "seriously: I don't know what happened, because it isn't "
                  "even documented by NVIDIA.";
    return ret;
  }
}

/*
 * returns the type of error
 * the different errors are in :
 * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9
 */
std::string yacx::detail::whichError(const CUresult &error) {
  std::string ret{"     Error: "};
  ret.append(std::to_string(error));
  ret.append("~");
  std::string description;
  switch (error) {
  case 0:
    ret = "NO_Error: 0~CUDA_SUCCESS";
    description = "The API call returned with no errors. In the case of query "
                  "calls, this also means that the operation being queried is "
                  "complete (see cuEventQuery() and cuStreamQuery()).";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 1:
    ret.append("CUDA_ERROR_INVALID_VALUE");
    description = "This indicates that one or more of the parameters passed to "
                  "the API call is not within an acceptable range of values.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 2:
    ret.append("CUDA_ERROR_OUT_OF_MEMORY");
    description = "The API call failed because it was unable to allocate "
                  "enough memory to perform the requested operation.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 3:
    ret.append("CUDA_ERROR_NOT_INITIALIZED");
    description =
        "This indicates that the CUDA driver has not been initialized with "
        "cuInit() or that initialization has failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 4:
    ret.append("CUDA_ERROR_DEINITIALIZED ");
    description = "This indicates that the CUDA driver is in the process of "
                  "shutting down.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 5:
    ret.append("CUDA_ERROR_PROFILER_DISABLED");
    description = "This indicates profiler is not initialized for this run. "
                  "This can happen when the application is running with "
                  "external profiling tools like visual profiler.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 6:
    ret.append("CUDA_ERROR_PROFILER_NOT_INITIALIZED");
    description =
        "Deprecated This error return is deprecated as of CUDA 5.0. It is no "
        "longer an error to attempt to enable/disable the profiling via "
        "cuProfilerStart or cuProfilerStop without initialization.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 7:
    ret.append("CUDA_ERROR_PROFILER_ALREADY_STARTED");
    description = "Deprecated This error return is deprecated as of CUDA 5.0. "
                  "It is no longer an error to call cuProfilerStart() when "
                  "profiling is already enabled.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 8:
    ret.append("CUDA_ERROR_PROFILER_ALREADY_STOPPED");
    description = "Deprecated This error return is deprecated as of CUDA 5.0. "
                  "It is no longer an error to call cuProfilerStop() when "
                  "profiling is already disabled.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 100:
    ret.append("CUDA_ERROR_NO_DEVICE");
    description = "This indicates that no CUDA-capable devices were detected "
                  "by the installed CUDA driver.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 101:
    ret.append("CUDA_ERROR_INVALID_DEVICE");
    description = "This indicates that the device ordinal supplied by the user "
                  "does not correspond to a valid CUDA device.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 200:
    ret.append("CUDA_ERROR_INVALID_IMAGE");
    description = "This indicates that the device kernel image is invalid. "
                  "This can also indicate an invalid CUDA module.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 201:
    ret.append("CUDA_ERROR_INVALID_CONTEXT");
    description =
        "This most frequently indicates that there is no context bound to the "
        "current thread. This can also be returned if the context passed to an "
        "API call is not a valid handle (such as a context that has had "
        "cuCtxDestroy() invoked on it). This can also be returned if a user "
        "mixes different API versions (i.e. 3010 context with 3020 API calls). "
        "See cuCtxGetApiVersion() for more details.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 202:
    ret.append("CUDA_ERROR_CONTEXT_ALREADY_CURRENT");
    description =
        "Deprecated This error return is deprecated as of CUDA 3.2. It is no "
        "longer an error to attempt to push the active context via "
        "cuCtxPushCurrent(). This indicated that the context being supplied as "
        "a parameter to the API call was already the active context.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 205:
    ret.append("CUDA_ERROR_MAP_FAILED");
    description = "This indicates that a map or register operation has failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 206:
    ret.append("CUDA_ERROR_UNMAP_FAILED");
    description =
        "This indicates that an unmap or unregister operation has failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 207:
    ret.append("CUDA_ERROR_ARRAY_IS_MAPPED");
    description = "This indicates that the specified array is currently mapped "
                  "and thus cannot be destroyed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 208:
    ret.append("CUDA_ERROR_ALREADY_MAPPED");
    description = "This indicates that the resource is already mapped.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 209:
    ret.append("CUDA_ERROR_NO_BINARY_FOR_GPU");
    description =
        "This indicates that there is no kernel image available that is "
        "suitable for the device. This can occur when a user specifies code "
        "generation options for a particular CUDA source file that do not "
        "include the corresponding device configuration.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 210:
    ret.append("CUDA_ERROR_ALREADY_ACQUIRED");
    description = "This indicates that a resource has already been acquired.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 211:
    ret.append("CUDA_ERROR_NOT_MAPPED");
    description = "This indicates that a resource is not mapped.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 212:
    ret.append("CUDA_ERROR_NOT_MAPPED_AS_ARRAY");
    description = "This indicates that a mapped resource is not available for "
                  "access as an array.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 213:
    ret.append("CUDA_ERROR_NOT_MAPPED_AS_POINTER");
    description = "This indicates that a mapped resource is not available for "
                  "access as a pointer.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 214:
    ret.append("CUDA_ERROR_ECC_UNCORRECTABLE");
    description = "This indicates that an uncorrectable ECC error was detected "
                  "during execution.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 215:
    ret.append("CUDA_ERROR_UNSUPPORTED_LIMIT");
    description = "This indicates that the CUlimit passed to the API call is "
                  "not supported by the active device.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 216:
    ret.append("CUDA_ERROR_CONTEXT_ALREADY_IN_USE");
    description = "This indicates that the CUcontext passed to the API call "
                  "can only be bound to a single CPU thread at a time but is "
                  "already bound to a CPU thread.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 217:
    ret.append("CUDA_ERROR_PEER_ACCESS_UNSUPPORTED");
    description = "This indicates that peer access is not supported across the "
                  "given devices.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 218:
    ret.append("CUDA_ERROR_INVALID_PTX ");
    description = "This indicates that a PTX JIT compilation failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 219:
    ret.append("CUDA_ERROR_INVALID_GRAPHICS_CONTEXT");
    description = "This indicates an error with OpenGL or DirectX context.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 220:
    ret.append("CUDA_ERROR_NVLINK_UNCORRECTABLE");
    description = "This indicates that an uncorrectable NVLink error was "
                  "detected during the execution.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 221:
    ret.append("CUDA_ERROR_JIT_COMPILER_NOT_FOUND");
    description =
        "This indicates that the PTX JIT compiler library was not found.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 300:
    ret.append("CUDA_ERROR_INVALID_SOURCE");
    description = "This indicates that the device kernel source is invalid.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 301:
    ret.append("CUDA_ERROR_FILE_NOT_FOUND");
    description = "This indicates that the file specified was not found.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 302:
    ret.append("CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND");
    description =
        "This indicates that a link to a shared object failed to resolve.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 303:
    ret.append("CUDA_ERROR_SHARED_OBJECT_INIT_FAILED");
    description =
        "This indicates that initialization of a shared object failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 304:
    ret.append("CUDA_ERROR_OPERATING_SYSTEM");
    description = "This indicates that an OS call failed.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 400:
    ret.append("CUDA_ERROR_INVALID_HANDLE");
    description =
        "This indicates that a resource handle passed to the API call was not "
        "valid. Resource handles are opaque types like CUstream and CUevent.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 401:
    ret.append("CUDA_ERROR_ILLEGAL_STATE");
    description = "This indicates that a resource required by the API call is "
                  "not in a valid state to perform the requested operation.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 500:
    ret.append("CUDA_ERROR_NOT_FOUND");
    description =
        "This indicates that a named symbol was not found. Examples of symbols "
        "are global/constant variable names, texture names, and surface names.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 600:
    ret.append("CUDA_ERROR_NOT_READY");
    description =
        "This indicates that asynchronous operations issued previously have "
        "not completed yet. This result is not actually an error, but must be "
        "indicated differently than CUDA_SUCCESS (which indicates completion). "
        "Calls that may return this value include cuEventQuery() and "
        "cuStreamQuery().";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 700:
    ret.append("CUDA_ERROR_ILLEGAL_ADDRESS");
    description = "While executing a kernel, the device encountered a load or "
                  "store instruction on an invalid memory address. This leaves "
                  "the process in an inconsistent state and any further CUDA "
                  "work will return the same error. To continue using CUDA, "
                  "the process must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 701:
    ret.append("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES");
    description =
        "This indicates that a launch did not occur because it did not have "
        "appropriate resources. This error usually indicates that the user has "
        "attempted to pass too many arguments to the device kernel, or the "
        "kernel launch specifies too many threads for the kernel's register "
        "count. Passing arguments of the wrong size (i.e. a 64-bit pointer "
        "when a 32-bit int is expected) is equivalent to passing too many "
        "arguments and can also result in this error.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 702:
    ret.append("CUDA_ERROR_LAUNCH_TIMEOUT");
    description =
        "This indicates that the device kernel took too long to execute. This "
        "can only occur if timeouts are enabled - see the device attribute "
        "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This "
        "leaves the process in an inconsistent state and any further CUDA work "
        "will return the same error. To continue using CUDA, the process must "
        "be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 703:
    ret.append("CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING");
    description = "This error indicates a kernel launch that uses an "
                  "incompatible texturing mode.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 704:
    ret.append("CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED");
    description = "This error indicates that a call to cuCtxEnablePeerAccess() "
                  "is trying to re-enable peer access to a context which has "
                  "already had peer access to it enabled.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 705:
    ret.append("CUDA_ERROR_PEER_ACCESS_NOT_ENABLED");
    description = "This error indicates that cuCtxDisablePeerAccess() is "
                  "trying to disable peer access which has not been enabled "
                  "yet via cuCtxEnablePeerAccess().";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 708:
    ret.append("CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE");
    description = "This error indicates that the primary context for the "
                  "specified device has already been initialized.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 709:
    ret.append("CUDA_ERROR_CONTEXT_IS_DESTROYED");
    description = "This error indicates that the context current to the "
                  "calling thread has been destroyed using cuCtxDestroy, or is "
                  "a primary context which has not yet been initialized.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 710:
    ret.append("CUDA_ERROR_ASSERT");
    description =
        "A device-side assert triggered during kernel execution. The context "
        "cannot be used anymore, and must be destroyed. All existing device "
        "memory allocations from this context are invalid and must be "
        "reconstructed if the program is to continue using CUDA.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 711:
    ret.append("CUDA_ERROR_TOO_MANY_PEERS");
    description = "This error indicates that the hardware resources required "
                  "to enable peer access have been exhausted for one or more "
                  "of the devices passed to cuCtxEnablePeerAccess().";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 712:
    ret.append("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED");
    description = "This error indicates that the memory range passed to "
                  "cuMemHostRegister() has already been registered.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 713:
    ret.append("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED");
    description =
        "This error indicates that the pointer passed to cuMemHostUnregister() "
        "does not correspond to any currently registered memory region.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 714:
    ret.append("CUDA_ERROR_HARDWARE_STACK_ERROR ");
    description =
        "While executing a kernel, the device encountered a stack error. This "
        "can be due to stack corruption or exceeding the stack size limit. "
        "This leaves the process in an inconsistent state and any further CUDA "
        "work will return the same error. To continue using CUDA, the process "
        "must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 715:
    ret.append("CUDA_ERROR_ILLEGAL_INSTRUCTION");
    description =
        "While executing a kernel, the device encountered an illegal "
        "instruction. This leaves the process in an inconsistent state and any "
        "further CUDA work will return the same error. To continue using CUDA, "
        "the process must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 716:
    ret.append("CUDA_ERROR_MISALIGNED_ADDRESS");
    description = "While executing a kernel, the device encountered a load or "
                  "store instruction on a memory address which is not aligned. "
                  "This leaves the process in an inconsistent state and any "
                  "further CUDA work will return the same error. To continue "
                  "using CUDA, the process must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 717:
    ret.append("CUDA_ERROR_INVALID_ADDRESS_SPACE");
    description =
        "While executing a kernel, the device encountered an instruction which "
        "can only operate on memory locations in certain address spaces "
        "(global, shared, or local), but was supplied a memory address not "
        "belonging to an allowed address space. This leaves the process in an "
        "inconsistent state and any further CUDA work will return the same "
        "error. To continue using CUDA, the process must be terminated and "
        "relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 718:
    ret.append("CUDA_ERROR_INVALID_PC");
    description =
        "While executing a kernel, the device program counter wrapped its "
        "address space. This leaves the process in an inconsistent state and "
        "any further CUDA work will return the same error. To continue using "
        "CUDA, the process must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 719:
    ret.append("CUDA_ERROR_LAUNCH_FAILED");
    description =
        "An exception occurred on the device while executing a kernel. Common "
        "causes include dereferencing an invalid device pointer and accessing "
        "out of bounds shared memory. Less common cases can be system specific "
        "- more information about these cases can be found in the system "
        "specific user guide. This leaves the process in an inconsistent state "
        "and any further CUDA work will return the same error. To continue "
        "using CUDA, the process must be terminated and relaunched.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 720:
    ret.append("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE");
    description =
        "This error indicates that the number of blocks launched per grid for "
        "a kernel that was launched via either cuLaunchCooperativeKernel or "
        "cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of "
        "blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or "
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number "
        "of multiprocessors as specified by the device attribute "
        "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 800:
    ret.append("CUDA_ERROR_NOT_PERMITTED");
    description =
        "This error indicates that the attempted operation is not permitted.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 801:
    ret.append("CUDA_ERROR_NOT_SUPPORTED");
    description = "This error indicates that the attempted operation is not "
                  "supported on the current system or device.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 802:
    ret.append("CUDA_ERROR_SYSTEM_NOT_READY");
    description = "This error indicates that the system is not yet ready to "
                  "start any CUDA work. To continue using CUDA, verify the "
                  "system configuration is in a valid state and all required "
                  "driver daemons are actively running. More information about "
                  "this error can be found in the system specific user guide.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 803:
    ret.append("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH");
    description = "This error indicates that there is a mismatch between the "
                  "versions of the display driver and the CUDA driver. Refer "
                  "to the compatibility documentation for supported versions.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 804:
    ret.append("CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE");
    description =
        "This error indicates that the system was upgraded to run with forward "
        "compatibility but the visible hardware detected by CUDA does not "
        "support this configuration. Refer to the compatibility documentation "
        "for the supported hardware matrix or ensure that only supported "
        "hardware is visible during initialization via the "
        "CUDA_VISIBLE_DEVICES environment variable.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 900:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED");
    description = "This error indicates that the operation is not permitted "
                  "when the stream is capturing.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 901:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED");
    description = "This error indicates that the current capture sequence on "
                  "the stream has been invalidated due to a previous error.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 902:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_MERGE");
    description = "This error indicates that the operation would have resulted "
                  "in a merge of two independent capture sequences.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 903:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_UNMATCHED");
    description = "This error indicates that the capture was not initiated in "
                  "this stream.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 904:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_UNJOINED");
    description = "This error indicates that the capture sequence contains a "
                  "fork that was not joined to the primary stream.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 905:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_ISOLATION");
    description =
        "This error indicates that a dependency would have been created which "
        "crosses the capture sequence boundary. Only implicit in-stream "
        "ordering dependencies are allowed to cross the boundary.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 906:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT");
    description = "This error indicates a disallowed implicit dependency on a "
                  "current capture sequence from cudaStreamLegacy.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 907:
    ret.append("CUDA_ERROR_CAPTURED_EVENT");
    description = "This error indicates that the operation is not permitted on "
                  "an event which was last recorded in a capturing stream.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 908:
    ret.append("CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD");
    description =
        "A stream capture sequence not initiated with the "
        "CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was "
        "passed to cuStreamEndCapture in a different thread.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  case 999:
    ret.append("CUDA_ERROR_UNKNOWN");
    description = "This indicates that an unknown internal error has occurred.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  default:
    ret.append("~error_unknown");
    description = "''It's a trap! Abort! We never left the CPU o(╥﹏╥)o'' But "
                  "seriously: I don't know what happened, because it isn't "
                  "even documented by NVIDIA.";
    // description is added to the return string
    description = descriptionFkt(description);
    ret.append("\n     ");
    ret.append(description);
    return ret;
  }
#pragma GCC diagnostic pop
}
