#!/bin/bash

docker build -f Dockerfile_cluster -t adaptive_masked_imprinting .
docker tag adaptive_masked_imprinting images.borgy.elementai.lan/tensorflow/adaptive_masked_imprinting
docker push images.borgy.elementai.lan/tensorflow/adaptive_masked_imprinting
chmod +x ../train.py