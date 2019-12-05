/*******************************************************************************
 * Copyright (c) 2019 Jona Abdinghoff
 *
 * LICENSE TEXT
 ******************************************************************************/

/*! \file
 *
 *   \brief C++ bindings to easily compile and execute CUDA kernels
 *   \author Jona Abdinghoff
 *
 *   \version 0.2.0
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

#include "cudaexecutor/Device.hpp"
#include "cudaexecutor/Headers.hpp"
#include "cudaexecutor/Exception.hpp"
#include "cudaexecutor/Logger.hpp"
#include "cudaexecutor/Options.hpp"
#include "cudaexecutor/Program.hpp"
#include "cudaexecutor/Source.hpp"
#include "cudaexecutor/KernelArgs.hpp"
#include "cudaexecutor/Kernel.hpp"
#include "cudaexecutor/KernelTime.hpp"
#include "cudaexecutor/util.hpp"

#include <vector_types.h>

