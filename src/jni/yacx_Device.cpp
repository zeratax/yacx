#include "yacx_Device.h"

#include "Handle.h"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Devices.hpp"
#include <cstring>

using yacx::Device, yacx::Devices;

jstring Java_yacx_Device_getName (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
    	CHECK_NULL(devicePtr, NULL)

        auto devicenameString = devicePtr->name();

        auto jdevicename = env->NewStringUTF(devicenameString.c_str());

        return jdevicename;
    END_TRY_R("getting name from Device", NULL);
}

jlong Java_yacx_Device_getMemorySize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto memory = devicePtr->total_memory();

        return memory;
    END_TRY_R("getting memory size from Device", 0);
}

jintArray Java_yacx_Device_getMaxBlock (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, NULL)

        dim3 block;
        devicePtr->max_block_dim(&block);

        auto res = env->NewIntArray(3);

        CHECK_NULL(res, NULL)

        int data[3];
        data[0] = block.x;
        data[1] = block.y;
        data[2] = block.z;

        env->SetIntArrayRegion(res, 0, 3,
                               reinterpret_cast<const jint*>(data));

        return res;
    END_TRY_R("getting maxBlockSize from Device", NULL);
}

jintArray Java_yacx_Device_getMaxGrid (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, NULL)

        dim3 grid;
        devicePtr->max_grid_dim(&grid);

        auto res = env->NewIntArray(3);
        if (res == nullptr) return nullptr;

        int data[3];
        data[0] = grid.x;
        data[1] = grid.y;
        data[2] = grid.z;

        env->SetIntArrayRegion(res, 0, 3,
                               reinterpret_cast<const jint*>(data));

        return res;
    END_TRY_R("getting maxGridSize from Device",NULL);
}

jint Java_yacx_Device_getMultiprocessorCount (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto multiprocessors = devicePtr->multiprocessor_count();

        return multiprocessors;
    END_TRY_R("getting number of Multiprocessors from Device", 0);
}

jint Java_yacx_Device_getClockRate (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto clockRate = devicePtr->clock_rate();

        return clockRate;
    END_TRY_R("getting clock rate from Device", 0);
}

jint Java_yacx_Device_getMemoryClockRate (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto memoryClockRate = devicePtr->memory_clock_rate();

        return memoryClockRate;
    END_TRY_R("getting memory clock rate from Device", 0);
}

jint Java_yacx_Device_getBusWidth (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto busWidth = devicePtr->bus_width();

        return busWidth;
    END_TRY_R("getting bus width from Device", 0);
}

jint Java_yacx_Device_getMinorVersion (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto minorVersion = devicePtr->minor_version();

        return minorVersion;
    END_TRY_R("getting minor version from Device", 0);
}


jint Java_yacx_Device_getMajorVersion (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto majorVersion = devicePtr->major_version();

        return majorVersion;
    END_TRY_R("getting major version from Device", 0);
}

jstring Java_yacx_Device_getUUID(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto uuidString = devicePtr->uuid();
        auto uuidJString = env->NewStringUTF(uuidString.c_str());

        return uuidJString;
    END_TRY_R("getting UUID from Device", 0);
}