#!/bin/bash 

#	rm -rf minicom.cap

#	minicom myusb0

# 	Ctrl/A L 

#	Before running need to run . ~/Ultibo_Projects/picoultibo.sh
	#This is need to set openocd to your path

#This file needs to copied to the build folder

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_mock_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/command_responder_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/feature_provider_mock_test.elf verify reset exit"

sleep 10



sleep 10


openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/keyword_benchmark/keyword_benchmark.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/greedy_memory_planner_test/greedy_memory_planner_test.elf verify reset exit"

sleep 10
openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_activations_test/kernel_activations_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/kernel_svdf_test/kernel_svdf_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program tests/recording_micro_allocator_test/recording_micro_allocator_test.elf verify reset exit"

sleep 10

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/hello_world/hello_world.elf verify reset exit"

sleep 10
echo "Tests Done"

#	kernel_svdf_test
# 	recording_micro_allocator_test