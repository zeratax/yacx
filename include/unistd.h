#pragma once

#ifdef _MSC_VER
#include <io.h>
#include <stdlib.h>

#define isatty _isatty
#define lseek _lseek

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#else
#include <unistd.h>
#endif