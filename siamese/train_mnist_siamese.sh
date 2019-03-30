#!/usr/bin/env sh
set -e

cd /opt/caffe

bash ./examples/siamese/model02/train_mnist_siamese.sh ./examples/siamese/model03/train_mnist_siamese.sh

