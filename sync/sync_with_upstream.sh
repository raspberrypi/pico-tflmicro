#!/bin/bash

UPSTREAM_REPO_URL=https://github.com/tensorflow/tflite-micro-arduino-examples

BUILD_DIR=${PWD}/build
ARDUINO_REPO_DIR=${BUILD_DIR}/arduino_repo
rm -rf ${ARDUINO_REPO_DIR}
mkdir -p ${BUILD_DIR}
git clone ${UPSTREAM_REPO_URL} ${ARDUINO_REPO_DIR} --depth=1
rm -rf src
mkdir src
cp -r ${ARDUINO_REPO_DIR}/src/tensorflow src/tensorflow
cp -r ${ARDUINO_REPO_DIR}/src/third_party src/third_party
cp sync/micro_time.cpp src/tensorflow/lite/micro/micro_time.cpp
cp sync/system_setup.cpp src/tensorflow/lite/micro/system_setup.cpp

rm -rf CMakeLists.txt
SOURCE_LIST_FILE="$(mktemp /tmp/source_file_list.XXXXXXXXX)" || exit 1
find src \( -iname "*.cpp" -o -iname "*.c" -o -iname "*.h" \) | \
  sort | \
  sed -E 's#.*#\  ${CMAKE_CURRENT_LIST_DIR}/\0#g' \
  > ${SOURCE_LIST_FILE}

sync/replace_string_with_file_contents.py \
  CMakeLists_template.txt \
  ${SOURCE_LIST_FILE} \
  "{{LIBRARY_SOURCES}}" \
  > CMakeLists.txt