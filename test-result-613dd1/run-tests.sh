#!/bin/bash 

#	rm -rf minicom.cap

#	minicom myusb0

# 	Ctrl/A L 

#	Before running need to run . ~/Ultibo_Projects/picoultibo.sh
	#This is need to set openocd to your path

#This file needs to copied to the build folder

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_mock_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/command_responder_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/feature_provider_mock_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/keyword_benchmark/keyword_benchmark.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/greedy_memory_planner_test/greedy_memory_planner_test.elf verify reset exit"

sleep 3
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_activations_test/kernel_activations_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_add_test/kernel_add_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_arg_min_max_test/kernel_arg_min_max_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_ceil_test/kernel_ceil_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_circular_buffer_test/kernel_circular_buffer_test.elf verify reset exit"

#sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_comparisons_test/kernel_comparisons_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_concatenation_test/kernel_concatenation_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_conv_test/kernel_conv_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_depthwise_conv_test/kernel_depthwise_conv_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_dequantize_test/kernel_dequantize_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_detection_postprocess_test/kernel_detection_postprocess_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_elementwise_test/kernel_elementwise_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_floor_test/kernel_floor_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_fully_connected_test/kernel_fully_connected_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_hard_swish_test/kernel_hard_swish_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_l2norm_test/kernel_l2norm_test.elf verify reset exit"

sleep 3
 
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_logical_test/kernel_logical_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_logistic_test/kernel_logistic_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_maximum_minimum_test/kernel_maximum_minimum_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_mul_test/kernel_mul_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_neg_test/kernel_neg_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_pad_test/kernel_pad_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_pack_test/kernel_pack_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_pooling_test/kernel_pooling_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_prelu_test/kernel_prelu_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_quantization_util_test/kernel_quantization_util_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_quantize_test/kernel_quantize_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_reduce_test/kernel_reduce_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_reshape_test/kernel_reshape_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_resize_nearest_neighbor_test/kernel_resize_nearest_neighbor_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_round_test/kernel_round_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_shape_test/kernel_shape_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_softmax_test/kernel_softmax_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_split_test/kernel_split_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_split_v_test/kernel_split_v_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_strided_slice_test/kernel_strided_slice_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_sub_test/kernel_sub_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_svdf_test/kernel_svdf_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_tanh_test/kernel_tanh_test.elf verify reset exit"
 
sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_unpack_test/kernel_unpack_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/linear_memory_planner_test/linear_memory_planner_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/memory_arena_threshold_test/memory_arena_threshold_test.elf verify reset exit"

sleep 3
#*********************************************
 
#~~~SOME TESTS FAILED~~

#*********************************************

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/memory_helpers_test/memory_helpers_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_allocator_test/micro_allocator_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_error_reporter_test/micro_error_reporter_test.elf verify reset exit"

sleep 3
             
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_interpreter_test/micro_interpreter_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_mutable_op_resolver_test/micro_mutable_op_resolver_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_string_test/micro_string_test.elf verify reset exit"

sleep 3
 
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_time_test/micro_time_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_utils_test/micro_utils_test.elf verify reset exit"

sleep 3
 
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/recording_micro_allocator_test/recording_micro_allocator_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/recording_simple_memory_allocator_test/recording_simple_memory_allocator_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/simple_memory_allocator_test/simple_memory_allocator_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/testing_helpers_test/testing_helpers_test.elf verify reset exit"

sleep 3



#openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/hello_world/hello_world.elf verify reset exit"

#sleep 3

#openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_error_reporter_test/micro_error_reporter_test.elf verify reset exit"

#sleep 3

#openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/micro_error_reporter_test/micro_error_reporter_test.elf verify reset exit"

#sleep 3

#43 additional test 66 as of these.
echo "Tests Done"
