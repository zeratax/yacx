checkCudaErrors(cuDeviceGet(device, 0));

char name[50];
// Returns an identifer string for the device.
cuDeviceGetName(name, 50, *device);
printf("> Using device 0: %s\n", name);

checkCudaErrors(cuDeviceGetAttribute(
    &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, *device));
checkCudaErrors(cuDeviceGetAttribute(
    &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, *device));
printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
