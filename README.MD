# CUDA EXECUTOR

wrapper to easily compile and execute cuda kernels

## Usage

### Example

```console
$ make example1
$ ./bin/runner
```

### Tests

```console
$ make check
```

For more info run `./bin/tester -?`

## Resources

- OpenCL to CUDA:
  - <https://www.sharcnet.ca/help/index.php/Porting_CUDA_to_OpenCL>
- DriverAPI & NVRTC:
  - <https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api>
  - <https://docs.nvidia.com/cuda/nvrtc/index.html>
- Similar Projects:
  - <https://github.com/mgopshtein/cudacpp/blob/master/examples/invoke-rtc.cpp>
  - <https://github.com/NVIDIA/jitify>
