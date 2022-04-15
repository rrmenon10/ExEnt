#!/usr/bin/env bash

benchmark_type=$1
config="config/clues_${benchmark_type}.json"
python src/train.py -c ${config}