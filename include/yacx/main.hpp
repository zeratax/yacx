/*******************************************************************************
 * Copyright (c) 2020 Jona Abdinghoff
 *
 * LICENSE TEXT
 ******************************************************************************/

/*! \file
 *
 *   \brief C++ bindings to easily compile and execute CUDA kernels
 *   \author Jona Abdinghoff
 *
 *   \version 0.6.1
 *
 */

/*! \mainpage
 * \section intro Introduction
 *
 * \section example Example
 *
 * The following example shows a general use case for the C++
 * bindings
 *
 * \include example_program.cpp
 * \example example_program.cpp
 * \endcode
 *
 */

#pragma once

#include "yacx/Devices.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Headers.hpp"
#include "yacx/Kernel.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/KernelTime.hpp"
#include "yacx/Logger.hpp"
#include "yacx/Options.hpp"
#include "yacx/Program.hpp"
#include "yacx/Source.hpp"
#include "yacx/util.hpp"

#include <vector_types.h>
