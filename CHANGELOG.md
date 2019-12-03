# Changelog
## cudaexecutor v0.2 (2019-12-03)
### Features
- c++ bindings
  - get devices
  - template kernels
  - logging and exception for debugging
  - nvrtc option class
- mostly replicated as a JNI

### Changes
- Classes renamed
  - ProgramArg => KernelArg
  - Program => Source
  - Kernel => Kernel, Program
  
## cudaexecutor v0.1 (2019-11-22)
### Features
- c++ bindings
  - execute abitrary cuda kernels
