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

namespace yacx {
extern bool gColoredOutput;
extern const char *gColorBrightRed;
extern const char *gColorBrightGreen;
extern const char *gColorBrightYellow;
extern const char *gColorBrightDefault;
extern const char *gColorGray;
extern const char *gColorReset;
} // namespace yacx
