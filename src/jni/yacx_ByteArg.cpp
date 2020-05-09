#include "yacx_ByteArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_ByteArg_createValue(JNIEnv* env, jclass cls, jbyte jvalue){
	BEGIN_TRY
		jclass clsKernelArg = getClass(env, "yacx/KernelArg");
		if (clsKernelArg == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{sizeof(jbyte), false, false, false, CTYPE};
		*(static_cast<jbyte*> (kernelArgPtr->getHostData())) = jvalue;

		return createJNIObject(env, clsKernelArg, kernelArgPtr);
	END_TRY_R("creating ByteValueArg", NULL)
}

jobject Java_yacx_ByteArg_createInternal (JNIEnv* env, jclass cls, jbyteArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayLength = env->GetArrayLength(jarray);
        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayLength * sizeof(jbyte), static_cast<bool>(jdownload), true, true, CTYPE + "*"};
        env->GetByteArrayRegion(jarray, 0, arrayLength, static_cast<jbyte*> (kernelArgPtr->getHostData()));

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY_R("creating ByteArg", NULL)
}

jbyteArray Java_yacx_ByteArg_asByteArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewByteArray(dataSize / sizeof(jbyte));

        CHECK_NULL(res, NULL)

        env->SetByteArrayRegion(res, 0, dataSize / sizeof(jbyte),
                                   reinterpret_cast<const jbyte*>(data));
        return res;
    END_TRY_R("getting ByteArg-content", NULL)
}
