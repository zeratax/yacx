#include "yacx_Devices.h"

#include "Handle.h"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Devices.hpp"
#include <cstring>

using yacx::Device, yacx::Devices;

jobject Java_yacx_Devices_findDevice__ (JNIEnv* env, jclass cls){
    BEGIN_TRY
        cls = getClass(env, "yacx/Device");
        CHECK_NULL(cls, NULL)

        Device* devicePtr = &Devices::findDevice();

    	return createJNIObject(env, cls, devicePtr);
    END_TRY_R("creating Device", NULL);
}

jobject Java_yacx_Devices_findDevice__Ljava_lang_String_2 (JNIEnv* env, jclass cls, jstring jdevicename){
    BEGIN_TRY
        cls = getClass(env, "yacx/Device");
        CHECK_NULL(cls, NULL)

        CHECK_NULL(jdevicename, NULL)

        auto devicenamePtr = env->GetStringUTFChars(jdevicename, NULL);

        Device* devicePtr = &Devices::findDevice(devicenamePtr);

        auto obj = createJNIObject(env, cls, devicePtr);

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        return obj;
    END_TRY_R("creating Device with specific name", NULL);
}

jobject Java_yacx_Devices_findDeviceByUUID (JNIEnv* env, jclass cls, jstring juuid){
    BEGIN_TRY
        cls = getClass(env, "yacx/Device");
        CHECK_NULL(cls, NULL)

        CHECK_NULL(juuid, NULL)

        auto uuidPtr = env->GetStringUTFChars(juuid, NULL);

        Device* devicePtr = &Devices::findDeviceByUUID(std::string(uuidPtr));

        auto obj = createJNIObject(env, cls, devicePtr);

        env->ReleaseStringUTFChars(juuid, uuidPtr);

        return obj;
    END_TRY_R("creating Device with specific name", NULL);
}

jobjectArray Java_yacx_Devices_findDevices (JNIEnv* env, jclass cls){
    BEGIN_TRY
        cls = getClass(env, "yacx/Device");
        CHECK_NULL(cls, NULL)

        auto deviceVector = Devices::findDevices();

        auto res = (jobjectArray) env->NewObjectArray(deviceVector.size(), cls, NULL);
        CHECK_NULL(res, NULL)

        for(int i = 0; i < deviceVector.size(); i++){
            auto jDevice = createJNIObject(env, cls, deviceVector[i]);
            env->SetObjectArrayElement(res, i, jDevice);
        }

        return res;
    END_TRY_R("creating Device with specific name", NULL);
}