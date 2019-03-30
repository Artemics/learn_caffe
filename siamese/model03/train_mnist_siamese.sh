#!/usr/bin/env sh
set -e

TOOLS=./build/tools
LOG=./examples/siamese/model03/log
FILEPATH="$(LOG)/exec_$(echo date +%Y%M%D-%H%M%S).txt"

$TOOLS/caffe train --solver=./examples/siamese/model03/mnist_siamese_solver.prototxt 2>&1 | tee $FILEPATH
