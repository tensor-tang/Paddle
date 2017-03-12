#!/bin/bash
set -e
layer_num=50
batch_size=64
use_dummy=0
use_mkldnn=1

./run.sh test $layer_num $batch_size $use_dummy $use_mkldnn

