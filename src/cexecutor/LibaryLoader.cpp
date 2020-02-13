#include "../../include/yacx/cexecutor/LibaryLoader.hpp"
#include "../../include/yacx/Logger.hpp"

#include <iostream>
#include <sstream>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h> 
#include <fcntl.h>

using yacx::loglevel, yacx::detail::dynop, yacx::detail::opfn;

void load_op(struct dynop *dest, const char *filename) {
    //open libary
    void* handle = dlopen((std::string("./") + filename).c_str(), RTLD_LAZY);
    char* error = dlerror();
    if (handle == NULL){
        if (error != NULL){
            throw std::runtime_error(std::string("error while opening libary with compiled function") + error);
        } else {
            throw std::runtime_error("error while opening libary with compiled function");
        }
    }

    //Search op-struct in libary
    void* op = dlsym(handle, "op");
    error = dlerror();
    if (op == NULL){
        dlclose(handle);
        
        if (error != NULL){
            throw std::runtime_error(std::string("error while searching \"op\" in libary with"
            "compiled function") + error);
        } else {
            throw std::runtime_error("error while searching \"op\" in libary with compiled function");
        }
    }

    //Save result in passed struct
    dest->op = ((struct opfn*) op)->op;
    dest->libhandle = handle;
}

void unload_op(struct dynop *op) {
    if (op == NULL){
        return;
    }

    //Close libary
    int result = dlclose(op->libhandle);
    char* error = dlerror();
    if (result != 0){
        if (error != NULL){
            logger(loglevel::ERROR) << "error while closing libary with compiled function: " << error;
        } else {
            logger(loglevel::ERROR) << "unknown error while closing libary with compiled function";
        }
    }

    //clear struct
    op->libhandle = NULL;
    op->op = NULL;
}