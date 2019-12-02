#ifndef HANDLE_H_
#define HANDLE_H_

jfieldID getHandleField(JNIEnv* env, jobject obj);

template <typename T>
T* getHandle(JNIEnv* env, jobject obj)
{
    auto handle = env->GetLongField(obj, getHandleField(env, obj));
    return reinterpret_cast<T*>(handle);
}

template <typename T>
void setHandle(JNIEnv* env, jobject obj, T* t)
{
    auto handle = reinterpret_cast<jlong>(t);
    env->SetLongField(obj, getHandleField(env, obj), handle);
}

void clearHandle(JNIEnv* env, jobject obj);

jobjectArray createStringArray(JNIEnv* env, const char** stringArray, int size);

#define BEGIN_TRY try {
#define END_TRY(message)                                                                                         \
     } catch (const std::exception &err) {                                                                       \
        jclass jClass = env->FindClass("ExecutorFailureException");                                              \
                                                                                                                 \
        if(!jClass) {                                                                                            \
            logger(cudaexecutor::loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";\
        }                                                                                                        \
                                                                                                                 \
        env->ThrowNew(jClass, (std::string("Executor failure while ") + message + ": " + err.what()).c_str());   \
    } catch (...) {                                                                                              \
        jclass jClass = env->FindClass("ExecutorFailureException");                                              \
                                                                                                                 \
        if(!jClass) {                                                                                            \
            logger(cudaexecutor::loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";\
        }                                                                                                        \
                                                                                                                 \
        env->ThrowNew(jClass, (std::string("Executor failure while ") + message).c_str());                       \
    }
#endif