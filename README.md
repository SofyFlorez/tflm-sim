# tflm-sim

Pure Python implementation of a TensorFlow Lite for Microcontrollers (https://github.com/tensorflow/tflite-micro) (TFLM) platform.

**WORK IN PROGRESS!**

## Introduction

When building a hardware implementation of TFLM, it's very useful to have a known-good implementation that can be used as a golden reference in hardware test benches. That is the purpose of this package.

TFLM already includes a model interpreter utility, class `tf.lite.Interpreter` (https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter), which is capable of running TFLM models.

You can use a `tf.lite.Interpreter` object as a golden reference for hardware development: Create a `tf.lite.Interpreter` object from a quantized TFLM model (as a `.tflm file`), feed it some input data tensor, and you will get not only the model's output but also the internal inter-layer feature tensors.

So why do we need a new Python reimplementation of the TFLM engine?

**First**, the source for all operators and quantization calculations is written in plain Python. This makes it easier to use as a reference for hardware development because plain Python is far easier to understand than the original C++ source of TFLM, especially for hardware engineers. Additionally, it serves as a good learning exercise for me and perhaps others, by having the TFLM model operation spelled out in a transparent language.

**Second**, `tflm-sim` incorporates a couple of features that are useful for debugging hardware implementations of TFLM:

* Single-stepping of operators at a pixel level.
* Separate execution of quantization, bias, and rescaling operations.

Other such features may be added in the future -- and that's another advantage of having your own pure Python model: it's much easier to modify your own version of `tflm-sim` than it would be to customize the `tf.lite.Interpreter` C++ code.


