#include "../../include/yacx/cexecutor/CProgram.hpp"
#include "../../include/yacx/Logger.hpp"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>

using yacx::loglevel, yacx::CProgram, yacx::Options, yacx::detail::load_op, yacx::detail::unload_op, yacx::detail::dynop;

int CProgram::id = 0;

Options createDefaultOptions(){
    Options defaultOptions = Options();
    defaultOptions.insert("-Wall");
    defaultOptions.insert("-Wextra");
    defaultOptions.insert("--pedantic");
    return defaultOptions;
}

Options CProgram::DEFAULT_OPTIONS = createDefaultOptions();

CProgram::CProgram(const char* cProgram, const char* functionName, std::vector<std::string> &parameterTypes,
        const char *compiler, Options &options) {
    logger(loglevel::DEBUG) << "creating cProgram " << functionName << " with id: " << id
    << ", compiler: " << compiler;
    logger(loglevel::DEBUG1) << "cFunction:\n" << cProgram;

    id++;

    m_numberArguments = parameterTypes.size();

    //filename for srcFile
    std::stringstream srcFileS;
    srcFileS << "src_" << functionName << "_" << id << ".c";
    m_srcFile = srcFileS.str();

    //filename for libFile
    std::stringstream libFileS; 
    libFileS << "lib_" << functionName << "_" << id << ".so";
    m_libFile = libFileS.str();

    logger(loglevel::DEBUG1) << "compile it to " << m_srcFile << " and " << m_libFile;

    //compilerCommand with options
    std::stringstream compilerWithOptionsS;
    compilerWithOptionsS << compiler << " ";
    for (unsigned int i = 0; i < options.numOptions(); i++){
        compilerWithOptionsS << options.content()[i] << " ";
    }
    std::string compilerWithOptions = compilerWithOptionsS.str();

    //compile
    compile(cProgram, functionName, parameterTypes, compilerWithOptions);

    //open libary
    load_op(&m_op, m_libFile, std::string("op") + functionName);
}

CProgram::~CProgram() {
    logger(loglevel::DEBUG) << "destroy cProgram with id" << id;

    unload_op(&m_op);

    remove(m_srcFile.c_str());
    remove(m_libFile.c_str());
}

bool pointerArg(std::string &s){
    for (std::string::reverse_iterator rit=s.rbegin(); rit!=s.rend(); ++rit){
        if (*rit != ' '){
            if (*rit == '*'){
                return true;
            } else {
                return false;
            }
        }
    }

    throw std::runtime_error(std::string("invalid parameter type: ") + s);
}

void writeParam(std::vector<std::string> &parameterTypes, int i, std::ofstream& fileOut){
    fileOut << "\n        ";
    if (pointerArg(parameterTypes[i])){
        fileOut << "parameter[" << i << "]";
    } else {
        fileOut << "*((" << parameterTypes[i] << "*) " << "parameter[" << i << "])";
    }
}

void CProgram::createSrcFile(const char* cProgram, const char* functionName,
        std::vector<std::string> &parameterTypes, std::ofstream& fileOut) {
    std::string executeFunctionName("execute");
    executeFunctionName.append(functionName);
    std::string opfnName("opfn");
    opfnName.append(functionName);
    std::string opName("op");
    opName.append(functionName);

    //write c function to be executed
    fileOut << cProgram;
    fileOut << "\n\n";
    
    //function for start function to be executed
    fileOut << "void " << executeFunctionName << " (";
    fileOut << "void** parameter";
    fileOut << ") {\n";
    //run function to be executed
    fileOut << "    " << functionName << "(";
    unsigned int i = 0;
    for (; i < parameterTypes.size()-1; i++){
        writeParam(parameterTypes, i, fileOut);
        fileOut << ",";
    }
    writeParam(parameterTypes, i, fileOut);
    fileOut << ");\n";
    fileOut << "}\n\n";

    //struct to store function pointer
    fileOut << "struct " << opfnName << "{ void (*op)(void** parameter);};\n\n";
    
    //create instance of struct as interface
    fileOut << "struct " << opfnName << " " << opName << " = {.op = " << executeFunctionName << "};\n";
}

void CProgram::compile(const char* cProgram, const char* functionName, std::vector<std::string> &parameterTypes,
        std::string &compilerWithOptions) {
    logger(loglevel::DEBUG) << "creating source file...";

    //create and open output file
    std::ofstream fileOut;
    fileOut.open(m_srcFile);
    
    //write srcfile
    createSrcFile(cProgram, functionName, parameterTypes, fileOut);
    
    //close file
    fileOut.close();

    //command for compile file
    std::stringstream compilerCommand;
    compilerCommand << compilerWithOptions << " -fPIC -shared -Wl,-soname," << m_libFile
        << " -o " << m_libFile << " " << m_srcFile;

    logger(loglevel::DEBUG) << "compiling to dynamic libary: " << compilerCommand.str();

    //compile to libary
    std::string tmp = compilerCommand.str();
    int result = std::system(tmp.c_str());

    if (result != 0){
        throw std::runtime_error("compilation failed");
    }
}

void CProgram::execute(std::vector<void*> arguments) {
    logger(loglevel::DEBUG) << "execute CProgram: " << m_srcFile;

    if (arguments.size() != m_numberArguments){
        std::stringstream errorS;
        errorS << "invalid number of arguments expected: " << m_numberArguments << " found: " << arguments.size();
        throw std::runtime_error(errorS.str());
    }

    m_op.op(&arguments[0]);
}