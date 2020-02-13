#pragma once

#include "CProgram.hpp"
//#include "../../include/yacx/KernelArgs.hpp"

#include <vector> 

// void execute(CProgram* cProgram, std::vector<yacx::KernelArg> arguments);
void execute(CProgram* cProgram, std::vector<void*> arguments);