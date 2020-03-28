#pragma once

#include <unistd.h>

namespace yacx {
static bool gColoredOutput = isatty(STDOUT_FILENO);
static const char *gColorBrightRed = gColoredOutput ? "\u001b[31;1m" : "";
static const char *gColorBrightGreen = gColoredOutput ? "\u001b[32;1m" : "";
static const char *gColorBrightYellow = gColoredOutput ? "\u001b[35;1m" : "";
static const char *gColorBrightDefault = gColoredOutput ? "\u001b[39;1m" : "";
static const char *gColorGray = gColoredOutput ? "\u001b[100;1m" : "";
static const char *gColorReset = gColoredOutput ? "\u001b[0m" : "";
} // namespace yacx
