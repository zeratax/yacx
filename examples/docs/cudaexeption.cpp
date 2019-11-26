int nBytes = 10;
CUdeviceptr d_A;
try {
  checkCUresultError(cuMemAlloc(&d_A, (unsigned int)nBytes));
} catch (CUresultException<(CUresult)1> &e) { // CUDA_ERROR_INVALID_VALUE
  std::cout << "Wrong Exception caught" << std::endl;
  std::cout << e.what() << std::endl;
} catch (CUresultException<CUDA_ERROR_NOT_INITIALIZED> &e) {
  std::cout << "Correct Exception caught" << std::endl;
  std::cout << e.what() << std::endl;
} catch (std::exception &e) {
  // Other errors
  std::cout << "other Error\n";
  std::cout << e.what() << std::endl;
}