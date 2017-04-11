#!/bin/sh

function usage() {
    echo "run.sh topology(googlenet/alexnet/vgg/resnet) topology_version"
    echo "The topology_version is version in GoogleNet(only support v1 yet),\
 layer_num in VGG(16 or 19) and ResNet(50, 101 or 152)."
}

if [ $# -lt 1 ]; then
    echo "At least one input!"
    usage
    exit 0
fi

if [ $1 ]; then
    topology=$1
fi

if [ $2 ]; then
    topology_version=$2
fi

python paddle.py \
  --topology=$topology \
  --topology_version=$topology_version \
  --dot_period=1 \
  --log_period=1 \
  --num_passes=1 \
  --trainer_count=1 \
  --saving_period=1 \
  --local=0 \
  --use_gpu=0

