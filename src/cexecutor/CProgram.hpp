#pragma once

#include <fstream>

class CProgram {
    public:
        CProgram(char* cProgram, char* functionName, int numberParameters, char* compilerWithOptions);
        ~CProgram();

        const char* getLibFile() { return libFile; }
    private:
        void writeSrcFile(char* cProgram, char* functionName, int numberParameters, std::ofstream& fileOut);
        void compile(char* cProgram, char* functionName, int numberParameters, char* compilerWithOptions);

        static int id;
        const char* srcFile;
        const char* libFile;
};