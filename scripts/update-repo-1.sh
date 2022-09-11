#!/bin/bash

#-rw-r--r-- 1 devel devel 1072460 Sep 10 20:22 libpico-tflmicro.a
#-rw-r--r-- 1 devel devel  269688 Sep 10 20:19 libpico-tflmicro_test.a

diffuse ../CMakeLists.txt ../../../pico-tflmicro-090222/CMakeLists.txt &

cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/micro_utils.h ../src/tensorflow/lite/micro/kernels/micro_utils.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/mirror_pad.cc ../src/tensorflow/lite/micro/kernels/mirror_pad.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/mul.cc ../src/tensorflow/lite/micro/kernels/mul.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/mul.h ../src/tensorflow/lite/micro/kernels/mul.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/mul_common.cc ../src/tensorflow/lite/micro/kernels/mul_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/neg.cc ../src/tensorflow/lite/micro/kernels/neg.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/pack.cc ../src/tensorflow/lite/micro/kernels/pack.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/pad.cc ../src/tensorflow/lite/micro/kernels/pad.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/pooling.cc ../src/tensorflow/lite/micro/kernels/pooling.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/pooling.h ../src/tensorflow/lite/micro/kernels/pooling.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/pooling_common.cc ../src/tensorflow/lite/micro/kernels/pooling_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/prelu.cc ../src/tensorflow/lite/micro/kernels/pooling_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/prelu.cc ../src/tensorflow/lite/micro/kernels/prelu.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/prelu.h ../src/tensorflow/lite/micro/kernels/prelu.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/prelu_common.cc ../src/tensorflow/lite/micro/kernels/prelu_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/quantize.cc ../src/tensorflow/lite/micro/kernels/quantize.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/quantize.h ../src/tensorflow/lite/micro/kernels/quantize.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/quantize_common.cc ../src/tensorflow/lite/micro/kernels/quantize_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/read_variable.cc ../src/tensorflow/lite/micro/kernels/read_variable.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/reduce.cc ../src/tensorflow/lite/micro/kernels/reduce.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/reduce.h ../src/tensorflow/lite/micro/kernels/reduce.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/reduce_common.cc ../src/tensorflow/lite/micro/kernels/reduce_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/reshape.cc ../src/tensorflow/lite/micro/kernels/reshape.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/resize_bilinear.cc ../src/tensorflow/lite/micro/kernels/reduce_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc ../src/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/round.cc ../src/tensorflow/lite/micro/kernels/round.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/select.cc ../src/tensorflow/lite/micro/kernels/select.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/shape.cc ../src/tensorflow/lite/micro/kernels/shape.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/slice.cc ../src/tensorflow/lite/micro/kernels/slice.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/softmax.cc ../src/tensorflow/lite/micro/kernels/softmax.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/softmax.h ../src/tensorflow/lite/micro/kernels/softmax.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/softmax_common.cc ../src/tensorflow/lite/micro/kernels/softmax_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/space_to_batch_nd.cc ../src/tensorflow/lite/micro/kernels/space_to_batch_nd.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/space_to_depth.cc ../src/tensorflow/lite/micro/kernels/space_to_depth.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/split.cc ../src/tensorflow/lite/micro/kernels/split.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/split_v.cc ../src/tensorflow/lite/micro/kernels/split_v.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/squared_difference.cc ../src/tensorflow/lite/micro/kernels/squared_difference.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/squeeze.cc ../src/tensorflow/lite/micro/kernels/squeeze.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/strided_slice.cc ../src/tensorflow/lite/micro/kernels/strided_slice.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/sub.cc ../src/tensorflow/lite/micro/kernels/sub.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/sub.h ../src/tensorflow/lite/micro/kernels/sub.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/sub_common.cc ../src/tensorflow/lite/micro/kernels/sub_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/svdf.cc ../src/tensorflow/lite/micro/kernels/svdf.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/svdf.h ../src/tensorflow/lite/micro/kernels/svdf.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/svdf_common.cc ../src/tensorflow/lite/micro/kernels/svdf_common.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/tanh.cc ../src/tensorflow/lite/micro/kernels/tanh.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/transpose.cc ../src/tensorflow/lite/micro/kernels/transpose.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/transpose_conv.cc ../src/tensorflow/lite/micro/kernels/transpose_conv.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc ../src/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test_config.h ../src/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test_config.h
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/unpack.cc ../src/tensorflow/lite/micro/kernels/unpack.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/var_handle.cc ../src/tensorflow/lite/micro/kernels/var_handle.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/while.cc ../src/tensorflow/lite/micro/kernels/while.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/kernels/zeros_like.cc ../src/tensorflow/lite/micro/kernels/zeros_like.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_helpers.cc ../src/tensorflow/lite/micro/memory_helpers.cc
cp  ../../../pico-tflmicro-090222/src/tensorflow/lite/micro/memory_helpers.h ../src/tensorflow/lite/micro/memory_helpers.h
