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
 *   \version 0.1.0
 *   \date November 2019
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
 * \include[lineno] example_program.cpp
 * \example example_program.cpp
 * \endcode
 *
 */

#pragma once

#include "Device.hpp"
#include "Exception.hpp"
#include "Logger.hpp"
#include "Options.hpp"
#include "Program.hpp"
#include "Source.hpp"
#include "util.hpp"

#include <vector_types.h>

