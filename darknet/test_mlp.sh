#!/bin/sh
./darknet classifier test cfg/mnist_mlp.data cfg/mnist_mlp.cfg ./backup/mnist_mlp.weights
