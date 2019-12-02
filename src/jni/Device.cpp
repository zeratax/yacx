#include "Device.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Device.hpp"
#include <cstring>

using cudaexecutor::Device;

jobject Java_Device_createDevice (JNIEnv* env, jclass cls){
    BEGIN_TRY
        Device* devicePtr = new Device{};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, devicePtr);

        return obj;
    END_TRY("creating Device");
}

jobject Java_Device_createDeviceInternal (JNIEnv* env, jclass cls, jstring jdevicename){
    BEGIN_TRY
        auto devicenamePtr = env->GetStringUTFChars(jdevicename, nullptr);

        Device* devicePtr = new Device{devicenamePtr};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, devicePtr);

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        return obj;
    END_TRY("creating Device with specific name");
}

jstring Java_Device_getName (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);

        auto devicenameString = devicePtr->name();

        char devicenamePtr[devicenameString.size() + 1];
        strcpy(devicenamePtr, devicenameString.c_str());

        auto jdevicename = env->NewStringUTF(devicenamePtr);

        return jdevicename;
    END_TRY("getting name from Device");
}

jlong Java_Device_getMemorySize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);

        auto memory = devicePtr->total_memory();

        return memory;
    END_TRY("getting memory size from Device");
}

jintArray Java_Device_getMaxBlock (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);

        dim3 block;
        devicePtr->max_block_dim(&block);

        auto res = env->NewIntArray(3);
        if (res == nullptr) return nullptr;

        int data[3];
        data[0] = block.x;
        data[1] = block.y;
        data[2] = block.z;

        env->SetIntArrayRegion(res, 0, 3,
                               reinterpret_cast<const jint*>(data));

        return res;
    END_TRY("getting maxBlockSize from Device");
}

jintArray Java_Device_getMaxGrid (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto devicePtr = getHandle<Device>(env, obj);

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