# CS1430 Final Project: An Improved WebGazer Using Neural Networks 

We improve WebGazer by using a neural network with grayscale 24x24 pixel eye patches as input. The output is then scaled and combined with the pupil position from clmgazr to predict gaze. 

A nonfunctional, but running version of the model is included in the codebase.

## Project Structure

/build
* Contains the built javascript module
* Run ./build_library to build

/dependecies
* Stores libraries: clmtrackr, object detection, etc
* Util functions

/src
* Contains the Javascript source code that builds to /build

/tfCode
* Contains the various experiments using TensorFlow

/app
* Contains the full stack webapp that integrates the TensorFlow model with the WebGazer webpage
* See README contained in this directory for instructions on how to run it.

## Who We Are

* Nicolas Choi
* Anthony Daoud
* Miranda Chao
