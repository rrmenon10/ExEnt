#!/usr/bin/env bash

model_name=$1
benchmark_type=$2
config="config/${model_name}/clues_${benchmark_type}.json"
python src/train.py -c ${config}