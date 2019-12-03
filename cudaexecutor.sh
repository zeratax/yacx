#!/bin/bash

PWD=`pwd`
JAVA_BIN="build/java/bin"

buildj () {
    rm -rf build
    mkdir -p $JAVA_BIN
    cp examples/kernels/* $JAVA_BIN
    pushd build
    cmake ../
    make JExampleSaxpy

    popd
    echo 'Build finished.'
}

exej () {
    pushd $JAVA_BIN
    java -ea -Djava.library.path=../../ $1
    popd
}

builds () {
  buildj
  cp examples/scala/*.scala $JAVA_BIN
  pushd $JAVA_BIN
  scalac *.scala
  popd
  echo 'Build finished.'
}

exes () {
  pushd $JAVA_BIN
  scala -J-ea -Djava.library.path=../../ $1
  popd
}


if [ "$1" != "" ]; then
    while [ "$1" != "" ]; do
        case $1 in
            --buildj                 )   buildj
                                         ;;
            --execute-java | --exej  )   exej $2
                                         ;;
            --builds                 )   buildj
                                         ;;
            --execute-scala | --exes )   exej $2
        esac
        shift
    done
else
    echo
    echo 'cudaexecutor'
    echo 'Options: ./cudaexecutor.sh'
    echo '--buildj             Tries CMake build'
    echo '--execute-java       Tests library with java'
    echo '--builds             Tries CMake build'
    echo '--execute-scala      Tests library with scala'
    echo
fi



