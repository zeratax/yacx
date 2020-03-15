#pragma once

namespace yacx{
//! converts floats to halfs
//! \param floats pointer to floats
//! \param halfs pointer to allocated memory for converted halfs
//! \param length number of floats to convert
void convertFtoH(void *floats, void *halfs, unsigned int length);

//! converts halfs to floats
//! \param halfs pointer to halfs
//! \param floats pointer to allocated memory for converted floats
//! \param length number of halfs to convert
void convertHtoF(void *halfs, void *floats, unsigned int length);
}

//TODO
// //! converts floats to halfs and transposes the matrix
// //! \param floats pointer to floats
// //! \param halfs pointer to allocated memory for converted halfs
// //! \param columns amount of columns of the matrix
// //! \param length number of floats to convert
// void convertFtoHT(void *floats, void *halfs, int columns, unsigned int length);