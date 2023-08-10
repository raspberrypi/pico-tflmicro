#!/bin/bash -xe

UPSTREAM_REPO_URL=https://github.com/petewarden/tflite-micro-arduino-examples
ROOT_DIR=${PWD}
BUILD_DIR=${ROOT_DIR}/build
ARDUINO_REPO_DIR=${BUILD_DIR}/arduino_repo
rm -rf ${ARDUINO_REPO_DIR}
mkdir -p ${BUILD_DIR}
git clone ${UPSTREAM_REPO_URL} ${ARDUINO_REPO_DIR} --depth=1

rm -rf src
mkdir src
cp -r ${ARDUINO_REPO_DIR}/src/tensorflow src/tensorflow
cp -r ${ARDUINO_REPO_DIR}/src/third_party src/third_party
cp -r ${ARDUINO_REPO_DIR}/signal src/signal
cp sync/micro_time.cpp src/tensorflow/lite/micro/micro_time.cpp
cp sync/system_setup.cpp src/tensorflow/lite/micro/system_setup.cpp
cp sync/micro_profiler.cpp src/tensorflow/lite/micro/micro_profiler.cpp
#cp sync/arm_nn_mat_mult_nt_t_s8.c src/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
mkdir -p src/tensorflow/lite/micro/benchmarks
cp sync/micro_benchmark.h src/tensorflow/lite/micro/benchmarks

mkdir -p src/tensorflow/lite/micro/testing
cp sync/micro_test.h src/tensorflow/lite/micro/testing

rm -rf CMakeLists.txt
SOURCE_LIST_FILE="$(mktemp /tmp/source_file_list.XXXXXXXXX)" || exit 1
find src \( -iname "*.cpp" -o -iname "*.c" -o -iname "*.h" \) | \
  sort | \
  sed -E 's#(.*)#\  ${CMAKE_CURRENT_LIST_DIR}/\1#g' \
  > ${SOURCE_LIST_FILE}

sync/replace_string_with_file_contents.py \
  CMakeLists_template.txt \
  ${SOURCE_LIST_FILE} \
  "{{LIBRARY_SOURCES}}" \
  > CMakeLists.txt