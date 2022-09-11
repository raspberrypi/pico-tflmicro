#!/bin/bash

#-rw-r--r-- 1 devel devel 1072460 Sep 10 20:22 libpico-tflmicro.a
#-rw-r--r-- 1 devel devel  269688 Sep 10 20:19 libpico-tflmicro_test.a

diffuse ../CMakeLists.txt ../../../pico-tflmicro-090222/CMakeLists.txt &

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc ../src/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/greedy_memory_planner.h ../src/tensorflow/lite/micro/memory_planner/greedy_memory_planner.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc ../src/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/linear_memory_planner.h ../src/tensorflow/lite/micro/memory_planner/linear_memory_planner.h

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/micro_memory_planner.h ../src/tensorflow/lite/micro/memory_planner/micro_memory_planner.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc ../src/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.h ../src/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_planner/memory_plan_struct.h ../src/tensorflow/lite/micro/memory_planner/memory_plan_struct.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_allocation_info.cc ../src/tensorflow/lite/micro/micro_allocation_info.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_allocation_info.h ../src/tensorflow/lite/micro/micro_allocation_info.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_allocator.cc ../src/tensorflow/lite/micro/micro_allocator.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/maximum_minimum.cc ../src/tensorflow/lite/micro/kernels/maximum_minimum.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_profiler.cc ../src/tensorflow/lite/micro/micro_profiler.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_profiler.h ../src/tensorflow/lite/micro/micro_profiler.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_resource_variable.cc ../src/tensorflow/lite/micro/micro_resource_variable.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_resource_variable.h ../src/tensorflow/lite/micro/micro_resource_variable.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_time.cc ../src/tensorflow/lite/micro/micro_time.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_time.h ../src/tensorflow/lite/micro/micro_time.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_string.cc ../src/tensorflow/lite/micro/micro_string.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_string.h ../src/tensorflow/lite/micro/micro_string.h

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_utils.cc ../src/tensorflow/lite/micro/micro_utils.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_utils.h ../src/tensorflow/lite/micro/micro_utils.h

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/mock_micro_graph.cc ../src/tensorflow/lite/micro/mock_micro_graph.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/mock_micro_graph.h ../src/tensorflow/lite/micro/mock_micro_graph.h

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_allocator.cc ../src/tensorflow/lite/micro/micro_allocator.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/micro_allocator.h ../src/tensorflow/lite/micro/micro_allocator.h

mkdir ../src/tensorflow/lite/micro/models/

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/models/person_detect_model_data.cc  ../src/tensorflow/lite/micro/models/person_detect_model_data.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/models/person_detect_model_data.h ../src/tensorflow/lite/micro/models/person_detect_model_data.h