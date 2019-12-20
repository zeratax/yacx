#include "yacx_Device.h"
#include "Handle.h"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Device.hpp"
#include <cstring>

using yacx::Device;

jobject Java_yacx_Device_createDevice (JNIEnv* env, jclass cls){
    BEGIN_TRY
        Device* devicePtr = new Device{};

    	return createJNIObject(env, cls, devicePtr);
    END_TRY("creating Device");
}

jobject Java_yacx_Device_createDeviceInternal (JNIEnv* env, jclass cls, jstring jdevicename){
    BEGIN_TRY
        CHECK_NULL(jdevicename, NULL)

        auto devicenamePtr = env->GetStringUTFChars(jdevicename, NULL);

        Device* devicePtr = new Device{devicenamePtr};

        auto obj = createJNIObject(env, cls, devicePtr);

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        return obj;
    END_TRY("creating Device with specific name");
}

jstring Java_yacx_Device_getName (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
    	CHECK_NULL(devicePtr, NULL)

        auto devicenameString = devicePtr->name();

        char devicenamePtr[devicenameString.size() + 1];
        strcpy(devicenamePtr, devicenameString.c_str());

        auto jdevicename = env->NewStringUTF(devicenamePtr);

        return jdevicename;
    END_TRY("getting name from Device");
}

jlong Java_yacx_Device_getMemorySize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto memory = devicePtr->total_memory();

        return memory;
    END_TRY("getting memory size from Device");
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
    END_TRY("getting maxBlockSize from Device");
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
    END_TRY("getting maxGridSize from Device");
}

jint Java_yacx_Device_getMultiprocessorCount (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto multiprocessors = devicePtr->multiprocessor_count();

        return multiprocessors;
    END_TRY("getting number of Multiprocessors from Device");
}

jint Java_yacx_Device_getClockRate (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto clockRate = devicePtr->clock_rate();

        return clockRate;
    END_TRY("getting clock rate from Device");
}

jint Java_yacx_Device_getMemoryClockRate (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto memoryClockRate = devicePtr->memory_clock_rate();

        return memoryClockRate;
    END_TRY("getting memory clock rate from Device");
}

jint Java_yacx_Device_getBusWidth (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);
		CHECK_NULL(devicePtr, 0)

        auto busWidth = devicePtr->bus_width();

        return busWidth;
    END_TRY("getting bus width from Device");
}
