#!/usr/bin/env sh
set -e

cd $CAFFE_DIR
TOOLS=./build/tools
LOG=./examples/siamese/model02/log
FILEPATH="$(LOG)/exec_$(echo date +%Y%M%D-%H%M%S).txt"

$TOOLS/caffe train --solver=./examples/siamese/model02/mnist_siamese_solver.prototxt 2>&1 | tee $FILEPATH

cd $OLDPWD
