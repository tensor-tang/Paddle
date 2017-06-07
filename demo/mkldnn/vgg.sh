#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Input task: train/test/pretrain/time"
    exit 0
fi

export KMP_AFFINITY="verbose,granularity=fine,compact,1,0"

./run.sh vgg $@
