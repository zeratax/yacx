#!/usr/bin/env bash

PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BUILD_DIR="build"
JAVA_BIN="${BUILD_DIR}/java/bin"

buildj() {
  pushd $PWD
  mkdir -p $JAVA_BIN
  cp -R examples/kernels $JAVA_BIN
  cp examples/java/*.java $JAVA_BIN

  cmake -H. -B$BUILD_DIR
  make -C $BUILD_DIR JNIExampleClasses

  #pushd $JAVA_BIN
  #javac *.java -d $PWD -sourcepath $PWD
  #popd
  popd
  echo 'Build finished.'
}

exej() {
  if [ "$1" == "" ]; then
    echo "!! parameter needed, select one of the following"
    find examples/java -type f -iname "Example*.java" -exec basename '{}' \; | sed 's/\.java$//1'
  else
    pushd "${PWD}/${JAVA_BIN}"
    java -ea -Xmx8G -Djava.library.path=../../ $1
    popd
  fi
}

builds() {
  buildj
  pushd $PWD
  cp examples/scala/*.scala $JAVA_BIN
  pushd $JAVA_BIN
  scalac *.scala
  popd
  popd
  echo 'Build finished.'
}

exes() {
  if [ "$1" == "" ]; then
    echo "!! parameter needed, select one of the following"
    find examples/scala -type f -iname "Example*.scala" -exec basename '{}' \; | sed 's/\.scala$//1'
  else
    pushd "${PWD}/${JAVA_BIN}"
    scala -J-ea -J-Xmx8G -Djava.library.path=../../ $1
    popd
  fi
}

if [ "$1" != "" ]; then
    case $1 in
    build-java) buildj;;
    execute-java) exej $2;;
    build-scala) builds;;
    execute-scala) exes $2;;
    esac
    shift
else
  echo 'yacx'
  echo 'Options: ./yacx.sh'
  echo 'build-java            Builds JNI and Java Classes'
  echo 'execute-java <class>  Execute Java Class'
  echo 'build-scala           Builds JNI, Java and Scala Classes'
  echo 'execute-scala <class> Execute Scala Class'
fi
