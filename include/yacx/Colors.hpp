#pragma once

#include <unistd.h>

namespace yacx {
bool gColoredOutput = isatty(STDOUT_FILENO);
const char *gColorBrightRed = gColoredOutput ? "\u001b[31;1m" : "";
const char *gColorBrightGreen = gColoredOutput ? "\u001b[32;1m" : "";
const char *gColorBrightYellow = gColoredOutput ? "\u001b[35;1m" : "";
const char *gColorBrightDefault = gColoredOutput ? "\u001b[39;1m" : "";
const char *gColorGray = gColoredOutput ? "\u001b[100;1m" : "";
const char *gColorReset = gColoredOutput ? "\u001b[0m" : "";
} // namespace yacx
