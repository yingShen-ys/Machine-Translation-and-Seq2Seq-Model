#!/bin/sh

conda env create -f environment.yml

conda activate pytorch_v0.3.1

conda install pytorch=0.3.1 cuda80 -c pytorch


