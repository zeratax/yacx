#!/bin/bash

PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BUILD_DIR="build"
JAVA_BIN="${BUILD_DIR}/java/bin"

buildj() {
  pushd $PWD
  mkdir -p $JAVA_BIN
  cp -R examples/kernels $JAVA_BIN
  cp examples/java/*.java $JAVA_BIN

  cmake -H. -B$BUILD_DIR
  make -C $BUILD_DIR yacx-jni

  pushd $JAVA_BIN
  javac *.java -d $PWD -sourcepath $PWD
  popd
  popd
  echo 'Build finished.'
}

exej() {
  if [ "$1" == "" ]; then
    echo "!! parameter needed, select one of the following"
    find examples/java -type f -iname "*.java" -exec basename '{}' \; | sed 's/\.java$//1'
  else
    pushd "${PWD}/${JAVA_BIN}"
    java -ea -Djava.library.path=../../ $1
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
    find examples/scala -type f -iname "*.scala" -exec basename '{}' \; | sed 's/\.java$//1'
  else
    pushd "${PWD}/${JAVA_BIN}"
    scala -J-ea -Djava.library.path=../../ $1
    popd
  fi
}

if [ "$1" != "" ]; then
    case $1 in
    --buildj)
      buildj
      ;;
    --execute-java | --exej)
      exej $2
      ;;
    --builds)
      builds
      ;;
    --execute-scala | --exes) exes $2 ;;
    esac
    shift
else
  echo
  echo 'yacx'
  echo 'Options: ./yacx.sh'
  echo '--buildj                Builds JNI and Java Classes'
  echo '--execute-java <class>  execute Java Class'
  echo '--builds                Builds JNI and Scala Classes'
  echo '--execute-scala <class> execute Scala Class'
  echo
fi
