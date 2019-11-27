#include "Program.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Source.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/Exception.hpp"

using cudaexecutor::loglevel, cudaexecutor::Source, cudaexecutor::Program, cudaexecutor::Kernel, cudaexecutor::nvrtcResultException;

jobject Java_Program_create (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName){
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

jobject Java_Program_compile (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programPtr = getHandle<Program>(env, obj);

        logger(loglevel::DEBUG1) << "est";
        Kernel* kernelPtr = new Kernel{programPtr->compile()};

        jclass jKernel = env->FindClass("Kernel");
        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;
    END_TRY("compiling kernel")
}