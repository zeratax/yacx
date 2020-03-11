#include "yacx_CProgram.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "KernelUtils.h"
#include "../../include/yacx/KernelArgs.hpp"
#include "../../include/yacx/Options.hpp"
#include "../../include/yacx/cexecutor/CProgram.hpp"

#include <iostream>
#include <sstream>

using yacx::KernelArg, jni::KernelArgJNI, yacx::Options, yacx::CProgram;

jobjectArray Java_yacx_CProgram_getTypes (JNIEnv* env, jclass, jobjectArray jKernelArgs){
    BEGIN_TRY
        auto argsLength = env->GetArrayLength(jKernelArgs);
        CHECK_BIGGER(argsLength, 0, "illegal array length", NULL)

        const char** types = new const char*[argsLength];

        for(int i = 0; i < argsLength; i++){
            auto jkernelArg = env->GetObjectArrayElement(jKernelArgs, i);
            CHECK_NULL(jkernelArg, NULL)

            auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
            CHECK_NULL(kernelArgJNIPtr, NULL)

            types[i] = kernelArgJNIPtr->getType().c_str();
        }

        auto jStringArray = createJStringArray(env, types, argsLength);

        delete types;

        return jStringArray;
    END_TRY_R("getting types of KernelArgs", NULL)
}

jobject Java_yacx_CProgram_createInternal(JNIEnv* env, jclass cls, jstring jcProgram, jstring jcFunctionName,
  jobjectArray jCtypes, jstring jcompiler, jobject joptions){
    BEGIN_TRY
        CHECK_NULL(jcProgram, NULL)
        CHECK_NULL(jcFunctionName, NULL)
        CHECK_NULL(jcompiler, NULL)
        CHECK_NULL(joptions, NULL)

        auto cProgram = env->GetStringUTFChars(jcProgram, nullptr);
        auto cFunctionName = env->GetStringUTFChars(jcFunctionName, nullptr);
        auto cTypes = jStringsToVector(env, jCtypes);
        auto compiler = env->GetStringUTFChars(jcompiler, nullptr);

        if (cTypes.empty()) return NULL;

        auto optionsPtr = getHandle<Options>(env, joptions);
        CHECK_NULL(optionsPtr, NULL);

        CProgram* cProgramPtr = new CProgram{cProgram, cFunctionName, cTypes, compiler, *optionsPtr};

        env->ReleaseStringUTFChars(jcProgram, cProgram);
        env->ReleaseStringUTFChars(jcFunctionName, cFunctionName);
        env->ReleaseStringUTFChars(jcompiler, compiler);

        return createJNIObject(env, cls, cProgramPtr);
    END_TRY_R("creating cProgram", NULL)
}

void Java_yacx_CProgram_execute(JNIEnv* env, jobject obj, jobjectArray jKernelArgs){
    BEGIN_TRY
        CHECK_NULL(jKernelArgs, )

        auto cProgramPtr = getHandle<CProgram>(env, obj);
        CHECK_NULL(cProgramPtr, )

        auto argumentsLength = env->GetArrayLength(jKernelArgs);
        CHECK_BIGGER(argumentsLength, 0, "illegal array length", )

        std::vector<void*> arguments;
        arguments.resize(argumentsLength);

        for(int i = 0; i < argumentsLength; i++){
            auto jkernelArg = env->GetObjectArrayElement(jKernelArgs, i);
            CHECK_NULL(jkernelArg, )

            auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
            CHECK_NULL(kernelArgJNIPtr, )

            arguments[i] = kernelArgJNIPtr->getHostData();
        }

        cProgramPtr->execute(arguments);
    END_TRY("executing cProgram")
}
