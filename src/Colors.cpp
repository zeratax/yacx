#include "yacx/Colors.hpp"

using yacx::gColoredOutput;

bool yacx::gColoredOutput = isatty(STDOUT_FILENO);
const char *yacx::gColorBrightRed = gColoredOutput ? "\u001b[31;1m" : "";
const char *yacx::gColorBrightGreen = gColoredOutput ? "\u001b[32;1m" : "";
const char *yacx::gColorBrightYellow = gColoredOutput ? "\u001b[33;1m" : "";
const char *yacx::gColorBrightDefault = gColoredOutput ? "\u001b[39;1m" : "";
const char *yacx::gColorGray = gColoredOutput ? "\u001b[90;1m" : "";
const char *yacx::gColorReset = gColoredOutput ? "\u001b[0m" : "";