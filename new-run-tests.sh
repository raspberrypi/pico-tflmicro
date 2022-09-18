#!/bin/bash
# This script needs to be run from the build folder.
#	cp ../new-run-tests.sh .
#	./new-run-tests.sh

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/hello_world/hello_world.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/micro_speech/audio_provider_mock_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/magic_wand/magic_wand.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/magic_wand/magic_wand_test.elf verify reset exit"

sleep 3

openocd -f interface/raspberrypi-swd.cfg -f target/rp2040.cfg -c "program examples/magic_wand/gesture_output_handler_test.elf verify reset exit"

sleep 3
