#!/bin/bash
docker run --rm --network host -v $PWD/zeppelin_notebook:/zeppelin/notebook --name zeppelin apache/zeppelin:0.9.0

