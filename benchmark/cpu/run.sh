#!/bin/bash
set -e

# settings
threads_num=$(grep 'processor' /proc/cpuinfo | sort -u | wc -l)
total_num=$(($threads_num>0?$threads_num:1))

use_dummy=1
use_mkldnn=0
use_mkldnn_wgt=0
version=
use_gpu=0
trainer_count=$total_num
cur_dir=$(cd "`dirname $0`"; pwd -P)
log_prefix=$cur_dir
dot_period=1
log_period=10
test_period=100

batch_sizes=(32 64 128 256 512 1024)


function usage() {
    echo "time_topology vgg/googlene/resnet version/layer_num"
    echo "The input (version/layer_num) is version in GoogleNet(only support v1 yet),\
 layer_num in VGG(16 or 19) and ResNet(50, 101 or 152)."
}

function time_topology() {
    if [ $# -lt 2 ]; then
        echo "At least 2 inputs!"
        usage
        exit 0
    fi
    topology=$1
    version=$2
    if [ $topology != "googlenet" ] && [ $topology != "vgg" ] \
     && [ $topology != "resnet" ]; then
        echo "unknown topology: ${topology}"
        usage
        exit 0
    fi
    echo "Start Timing ${topology}_${version} ......"
    topology="./${topology}.sh"
    for bs in ${batch_sizes[@]}; do
        # check trainer_count
        if [ $total_num -gt $bs ]; then
            trainer_count=$bs
        fi
        echo "use trainer_count=$trainer_count at batchsize=$bs"
        $topology time $bs $use_dummy $use_mkldnn $use_mkldnn_wgt \
$version $use_gpu $trainer_count $log_prefix $dot_period $log_period $test_period
        sleep 10s
    done
}

# main
clear
echo "-------------------------------------------------------------------------------"
cd ../../demo/mkldnn

# VGG
time_topology vgg 19
time_topology vgg 16

# GoogleNet v1
time_topology googlenet v1

# ResNet
time_topology resnet 50
time_topology resnet 101
time_topology resnet 152

cd -