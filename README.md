
# TensorFlow Lite Micro

An Open Source Machine Learning Framework for Everyone.

## Introduction

This is a version of the [TensorFlow Lite Micro library](https://www.tensorflow.org/lite/microcontrollers)
for the Raspberry Pi Pico microcontroller. It allows you to run machine learning models to
do things like voice recognition, detect people in images, recognize gestures from an accelerometer,
and other sensor analysis tasks.

## Getting Started

First you'll need to follow the Pico setup instructions to initialize the development
environment on your machine. Once that is done, make sure that the PICO_SDK_PATH
environment variable has been set to the location of the Pico SDK, either in the shell
you're building in, or the CMake configure environment variable setting of the extension
if you're using VS Code.

You should then be able to build the library, tests, and examples. The easiest way to
build is using VS Code's CMake integration, by loading the project and choosing the
build option at the bottom of the window.

## What's Included

There are several example applications included. The simplest one to begin with is the
hello_world project. This demonstrates the fundamentals of deploying an ML model on a
device, driving the Pico's LED in a learned sine-wave pattern.

Other examples include simple speech recognition, a magic wand gesture recognizer,
and spotting people in camera images, but because they require audio, accelerometer or
image inputs you'll need to write some code to hook up your own sensors, since these
are not included with the base microcontroller.

## Contributing

This repository (https://github.com/raspberrypi/pico-tflmicro) is read-only, because
it has been automatically generated from the master TensorFlow repository at
https://github.com/tensorflow/tensorflow. This means that all issues and pull requests
need to be filed there. You can generate a version of this generated project by
running the commands:

```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
tensorflow/lite/micro/tools/project/generate.py rp2 pico-tflmicro
```

This should create a Pico-compatible project from the latest version of the TensorFlow
repository.

## Learning More

The [TensorFlow website](https://www.tensorflow.org/lite/microcontrollers) has
information on training, tutorials, and other resources.

The [TinyML Book](https://tinymlbook.com) is a guide to using TensorFlow Lite Micro
across a variety of different systems.

[TensorFlowLite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/pdf/2010.08678.pdf)
has more details on the design and implementation of the framework.

## Licensing

The TensorFlow source code is covered by the license described in src/tensorflow/LICENSE,
components from other libraries have the appropriate licenses included in their
third_party folders.

