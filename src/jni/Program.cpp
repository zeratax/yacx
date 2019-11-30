#include "Program.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Headers.hpp"
#include "../../include/cudaexecutor/Source.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/Options.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"

using cudaexecutor::loglevel, cudaexecutor::Source, cudaexecutor::Program, cudaexecutor::Headers, cudaexecutor::Options, cudaexecutor::Kernel;

jobject Java_Program_createInternal__Ljava_lang_String_2Ljava_lang_String_2 (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName){
    BEGIN_TRY
        auto kernelSourcePtr = env->GetStringUTFChars(jkernelSource, nullptr);
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        Source source{kernelSourcePtr};
        Program* programPtr = new Program{source.program(kernelNamePtr)};

        env->ReleaseStringUTFChars(jkernelSource, kernelSourcePtr);
        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programPtr);

        return obj;
    END_TRY("creating program")
}

jobject Java_Program_createInternal__Ljava_lang_String_2Ljava_lang_String_2LHeaders_2 (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName, jobject jheaders){
    BEGIN_TRY
        auto kernelSourcePtr = env->GetStringUTFChars(jkernelSource, nullptr);
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        auto headersPtr = getHandle<Headers>(env, jheaders);

        Source source{kernelSourcePtr, *headersPtr};
        Program* programPtr = new Program{source.program(kernelNamePtr)};

        env->ReleaseStringUTFChars(jkernelSource, kernelSourcePtr);
        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programPtr);

        return obj;
    END_TRY("creating program")
}

jobject Java_Program_compile (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programPtr = getHandle<Program>(env, obj);

        Kernel* kernelPtr = new Kernel{programPtr->compile()};

        jclass jKernel = env->FindClass("Kernel");
        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;
    END_TRY("compiling kernel")
}

jobject Java_Program_compileInternal(JNIEnv* env, jobject obj, jobject joptions){
    BEGIN_TRY
        auto programPtr = getHandle<Program>(env, obj);
        const auto optionsPtr = getHandle<Options>(env, joptions);

        Kernel* kernelPtr = new Kernel{programPtr->compile(*optionsPtr)};

        jclass jKernel = env->FindClass("Kernel");
        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;

    END_TRY("compiling kernel")


}