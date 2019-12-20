#include "yacx_Program.h"
#include "Handle.h"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Headers.hpp"
#include "../../include/yacx/Source.hpp"
#include "../../include/yacx/Program.hpp"
#include "../../include/yacx/Options.hpp"
#include "../../include/yacx/Kernel.hpp"

using yacx::Source, yacx::Program, yacx::Headers, yacx::Options, yacx::Kernel;

jobject Java_yacx_Program_createInternal__Ljava_lang_String_2Ljava_lang_String_2 (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName){
    BEGIN_TRY
        CHECK_NULL(jkernelSource, NULL)
        CHECK_NULL(jkernelName, NULL)

        auto kernelSourcePtr = env->GetStringUTFChars(jkernelSource, nullptr);
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        Source source{kernelSourcePtr};
        Program* programPtr = new Program{source.program(kernelNamePtr)};

        env->ReleaseStringUTFChars(jkernelSource, kernelSourcePtr);
        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        return createJNIObject(env, cls, programPtr);
    END_TRY("creating program")
}

jobject Java_yacx_Program_createInternal__Ljava_lang_String_2Ljava_lang_String_2Lyacx_Headers_2 (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName, jobject jheaders){
    BEGIN_TRY
        CHECK_NULL(jkernelSource, NULL)
        CHECK_NULL(jkernelName, NULL)
        CHECK_NULL(jheaders, NULL)

        auto kernelSourcePtr = env->GetStringUTFChars(jkernelSource, nullptr);
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        auto headersPtr = getHandle<Headers>(env, jheaders);
        CHECK_NULL(headersPtr, NULL);

        Source source{kernelSourcePtr, *headersPtr};
        Program* programPtr = new Program{source.program(kernelNamePtr)};

        env->ReleaseStringUTFChars(jkernelSource, kernelSourcePtr);
        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        return createJNIObject(env, cls, programPtr);
    END_TRY("creating program")
}

jobject Java_yacx_Program_compile (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programPtr = getHandle<Program>(env, obj);
    	CHECK_NULL(programPtr, NULL);

        Kernel* kernelPtr = new Kernel{programPtr->compile()};

        jclass jKernel = getClass(env, "yacx/Kernel");
        if (jKernel == NULL) return NULL;

        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;
    END_TRY("compiling kernel")
}

jobject Java_yacx_Program_compileInternal(JNIEnv* env, jobject obj, jobject joptions){
    BEGIN_TRY
        CHECK_NULL(joptions, NULL)

        auto programPtr = getHandle<Program>(env, obj);
    	CHECK_NULL(programPtr, NULL);
        const auto optionsPtr = getHandle<Options>(env, joptions);
        CHECK_NULL(optionsPtr, NULL);

        Kernel* kernelPtr = new Kernel{programPtr->compile(*optionsPtr)};

        jclass jKernel = getClass(env, "yacx/Kernel");
        if (jKernel == NULL) return NULL;

        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;

    END_TRY("compiling kernel")
}
