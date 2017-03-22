#!/bin/bash
set -e

unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

function run() {
    task=$1
    topology=$2
    layer_num=$3
    bs=$4
    cfg=${topology}.py
    model=./models/${topology}_${layer_num}/
    output=./models/${topology}_${layer_num}
    thread=1
    is_test=0
    use_mkldnn=1
    use_mkldnn_wgt=0
    use_gpu=1
    if [ $5 ]; then
        thread=$5
    fi
    if [ $6 ]; then
        use_gpu=$6
    fi
    if [ $task == "test" ]; then
        is_test=1
        use_mkldnn_wgt=0 # the models are trained by CPU yet, so compatible with CPU weight
        if [ ! -d $model ]; then
            echo "$model does not exist!"
            exit 0
        fi
    fi
    if [ $thread -gt 1 ]; then
        use_mkldnn=0
        unset OMP_NUM_THREADS MKL_NUM_THREADS
        log=logs/"log_${task}_${topology}_${layer_num}_bs${bs}_thread${thread}.log"
    else
        log=logs/"log_${task}_${topology}_${layer_num}_bs${bs}_dummy.log"
    fi
    if [ -f $log ]; then
        echo "remove old log $log"
        rm -f $log
    fi
    if [ $use_gpu -eq 1 ] || [ $use_gpu == "True" ] ; then
        unset OMP_NUM_THREADS MKL_NUM_THREADS
        use_mkldnn=0
    fi
    if [ $use_mkldnn -eq 1 ] && [ $task == "train" ]; then
        use_mkldnn_wgt=1
    fi
    args="layer_num=${layer_num},batch_size=${bs},use_dummy=1,\
use_mkldnn=${use_mkldnn},use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test}"

    if [ $task == "time" ]; then
        paddle train --job=$task \
            --config=$cfg \
            --use_gpu=$use_gpu \
            --trainer_count=$thread \
            --log_period=10 \
            --test_period=100 \
            --config_args=$args \
            2>&1 | tee -a $log 2>&1 
    elif [ $task == "test" ]; then # test
        paddle train --job=$task \
            --config=$cfg \
            --use_gpu=$use_gpu \
            --trainer_count=$thread \
            --dot_period=1 \
            --log_period=10 \
            --init_model_path=$model \
            --config_args=$args \
            2>&1 | tee -a $log 2>&1 
    else
        paddle train --job=$task \
            --config=$cfg \
            --use_gpu=$use_gpu \
            --trainer_count=$thread \
            --dot_period=1 \
            --log_period=5 \
            --test_all_data_in_one_period=0 \
            --num_passes=2 \
            --save_dir=$output \
            --config_args=$args \
            2>&1 | tee -a $log 2>&1
    fi
}

train_list="data/train.list"
test_list="data/test.list"
if [ ! -d "data" ]; then
    mkdir -p data
fi
if [ ! -f $test_list ]; then
    echo " " > $test_list
fi
if [ ! -f $train_list ]; then
    echo " " > $train_list
fi

if [ ! -d "logs" ]; then
  mkdir logs
fi

### test
# GoogleNet
run test vgg 19 2
run test vgg 19 8
run test vgg 19 16
run test vgg 19 32
run test vgg 19 64

# VGG
run test resnet 50 2
run test resnet 50 8
run test resnet 50 16
run test resnet 50 32
run test resnet 50 64

# ResNet
run test googlenet v1 2
run test googlenet v1 8
run test googlenet v1 16
run test googlenet v1 32
run test googlenet v1 64

### time
# GoogleNet
run time googlenet v1 32
run time googlenet v1 64
run time googlenet v1 128
run time googlenet v1 256

# VGG
run time vgg 19 32
run time vgg 19 64
run time vgg 19 128
run time vgg 19 256

# ResNet
run time resnet 50 32
run time resnet 50 64
run time resnet 50 128
run time resnet 50 256

