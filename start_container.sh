#!/bin/bash

docker run --rm --gpus all -v $HOME/Gen_Algorithm_TFG:/tf -p 8888:8888 -p 8088:8088 --name gen_alg_cont tensorflow/tensorflow:latest-gpu-py3-jupyter
