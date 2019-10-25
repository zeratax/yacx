# CUDA COMPILER

compiles any cuda kernel

## Build

```shell
make
```

## Usage

```shell
/bin/runner --help
Program Usage:
  --help                produce help message
  -k [ --kernel ] arg   path to cuda kernel
  -o [ --options ] arg  compile options
  -h [ --header ] arg   cuda kernel headers
```

### Example

```shell
./bin/runner --kernel kernel/saxpy.cu -h foo.h -h bar.h -o "--fmad=false"
```
