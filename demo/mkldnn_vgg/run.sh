#!/bin/bash
set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

function usage() {
    echo "run.sh train/test/pretrain/time (bs) (use_dummy) (layer_num)\
 (use_mkldnn) (use_mkldnn_wgt)"
}

if [ $# -lt 1 ]; then
    echo "At least one input"
    usage
    exit 0
fi

cfg=vgg.py
topology="vgg"
train_list="data/train.list"
test_list="data/test.list"
if [ ! -d "data" ]; then
    mkdir -p data
fi

## inputs:
task=$1
bs=64
use_dummy=1
layer_num=19
use_mkldnn=1
use_mkldnn_wgt=1
is_test=0
if [ $task == "test" ]; then
    is_test=1
    use_mkldnn_wgt=0 # the models are trained by CPU yet, so compatible with CPU weight
fi
if [ $2 ]; then
    bs=$2
fi
if [ $3 ]; then
    use_dummy=$3
fi
if [ $4 ]; then
    layer_num=$4
fi
if [ $5 ]; then
    use_mkldnn=$5
fi
if [ $6 ]; then
    use_mkldnn_wgt=$6
fi

output=./models/${topology}_${layer_num}
model=models/${topology}_${layer_num}/pass-00001/
if [ $use_dummy -eq 1 ]; then
    log="log_${task}_${topology}_${layer_num}_bs${bs}_dummy.log"
else
    log="log_${task}_${topology}_${layer_num}_bs${bs}_image.log"
fi
if [ -f $log ]; then
    echo "remove old log $log"
    rm -f $log
fi
if [ $task == "train" ] || [ $task == "pretrain" ]; then
    if [ ! -f $train_list ]; then
        if [ $use_dummy -eq 1 ]; then
            echo " " > $train_list
        else
            echo "$train_list does not exist!"
            exit 0
        fi
    fi
fi
if [ ! -f $test_list ]; then
    if [ $use_dummy -eq 1 ]; then
        echo " " > $test_list
    else
        echo "$test_list does not exist!"
        exit 0
    fi
fi
if [ $is_test -eq 1 ] || [ $task == "pretrain" ]; then
    if [ ! -d $model ]; then
      echo "$model does not exist!"
      exit 0
    fi
fi

args="layer_num=${layer_num},batch_size=${bs},use_dummy=${use_dummy},use_mkldnn=${use_mkldnn},\
use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test}"

if [ $task == "train" ]; then
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --dot_period=1 \
    --log_period=2 \
    --test_all_data_in_one_period=0 \
    --trainer_count=1 \
    --num_passes=2 \
    --save_dir=$output \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1
elif [ $task == "time" ]; then
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --log_period=10 \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
else  # pretrain or test
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --log_period=1 \
    --init_model_path=$model \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
fi
