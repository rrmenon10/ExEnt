#!/bin/bash

if [ ! -d "env" ] ; then
    python -m venv env
    source env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

if [ ! -d "data" ] ; then
    wget https://cs.unc.edu/~rrmenon/data/clues.tar.gz
    tar -xvzf clues.tar.gz
    rm clues.tar.gz
fi