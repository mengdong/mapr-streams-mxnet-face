#!/usr/bin/env sh

export MXNET_ENGINE_TYPE=NaiveEngine
python -u detection.py --img face.jpeg --gpu 0 
