#!/bin/bash
set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

function usage() {
    echo "run.sh topology task (batch_size) (use_dummy)\
 (use_mkldnn) (use_mkldnn_wgt) (version/layer_num)"
    echo "Two required inputs: "
    echo "    topology: alexnet/googlenet/vgg/resnet."
    echo "    task    : train/test/pretrain/time."
    echo "The inputs with brackets are optional."
    echo "The last input is version in GoogleNet(only support v1 yet),\
 layer_num in VGG(16 or 19) and ResNet(50, 101 or 152)."
}

if [ $# -lt 2 ]; then
    echo "At least two input!"
    usage
    exit 0
fi

train_list="data/train.list"
test_list="data/test.list"
if [ ! -d "data" ]; then
    mkdir -p data
fi
if [ ! -d "models" ]; then
    mkdir -p models
fi

### required inputs ###
topology=$1
task=$2
# check inputs
if [ $topology != "googlenet" ] && [ $topology != "vgg" ] \
 && [ $topology != "resnet" ] && [ $topology != "alexnet" ]; then
    echo "unknown topology: ${topology}"
    usage
    exit 0
fi
if [ $task != "train" ] && [ $task != "time" ] \
 && [ $task != "test" ] && [ $task != "pretrain" ]; then
    echo "unknown task: ${task}"
    usage
    exit 0
fi

### other inputs ###
config="${topology}.py"
bs=64
use_dummy=1
use_mkldnn=1
use_mkldnn_wgt=1
version=
models_in=
models_out=
log_dir=
is_test=0
if [ $task == "test" ]; then
    is_test=1
    use_mkldnn_wgt=0 # suppose the models are trained by CPU, so compatible with CPU weight
fi
if [ $topology == "googlenet" ]; then
    version="v1"
elif [ $topology == "vgg" ]; then
    version=19
elif [ $topology == "resnet" ]; then
    version=50
fi
# default use mkldnn and mkldnn_wgt, and do not use dummy data when training
if [ $task == "train" ] || [ $task == "pretrain" ]; then
    use_dummy=0
    use_mkldnn=1
    use_mkldnn_wgt=1
fi
if [ $3 ]; then
    bs=$3
fi
if [ $4 ]; then
    use_dummy=$4
fi
if [ $5 ]; then
    use_mkldnn=$5
fi
if [ $6 ]; then
    use_mkldnn_wgt=$6
fi
if [ $7 ]; then
    version=$7
fi

# alexnet has not version option
if [ $topology == "alexnet" ]; then
    models_in=models/${topology}/pass-00001/
    models_out=./models/${topology}
    log_dir=logs/${topology}
else
    models_in=models/${topology}_${version}/pass-00001/
    models_out=./models/${topology}_${version}
    log_dir=logs/${topology}_${version}
fi
# prepare log
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi
if [ $use_dummy -eq 1 ]; then
    log="${log_dir}/log_${task}_${topology}_${version}_bs${bs}_dummy.log"
else
    log="${log_dir}/log_${task}_${topology}_${version}_bs${bs}_image.log"
fi
if [ -f $log ]; then
    echo "remove old $log"
    rm -f $log
fi
# check data
if [ $task == "train" ] || [ $task == "pretrain" ] || [ $task == "time" ]; then
    if [ ! -f $train_list ]; then
        if [ $use_dummy -eq 1 ]; then
            echo " " > $train_list
            made_train_list=1
        else
            echo "$train_list does not exist! task: $task"
            exit 0
        fi
    fi
fi
if [ ! -f $test_list ]; then
    if [ $use_dummy -eq 1 ]; then
        echo " " > $test_list
        made_test_list=1
    else
        echo "$test_list does not exist!  task: $task"
        exit 0
    fi
fi
if [ $is_test -eq 1 ] || [ $task == "pretrain" ]; then
    if [ ! -d $models_in ]; then
      echo "$models_in does not exist! task: $task"
      exit 0
    fi
fi

# init args
args="batch_size=${bs},use_dummy=${use_dummy},use_mkldnn=${use_mkldnn},\
use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test}"
if [ $topology == "googlenet" ]; then
    args="${args},version=${version}"
else
    args="${args},layer_num=${version}"
fi

# commands
if [ $task == "train" ]; then
    paddle train --job=$task \
    --config=$config \
    --use_gpu=False \
    --dot_period=1 \
    --log_period=1 \
    --test_all_data_in_one_period=0 \
    --trainer_count=1 \
    --num_passes=2 \
    --save_dir=$models_out \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1
elif [ $task == "time" ]; then
    paddle train --job=$task \
    --config=$config \
    --use_gpu=False \
    --log_period=10 \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
else  # pretrain or test
    paddle train --job=$task \
    --config=$config \
    --use_gpu=False \
    --dot_period=1 \
    --log_period=5 \
    --init_model_path=$models_in \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
fi

# clean lists
if [ $made_train_list ] && [ -f $train_list ]; then
    rm -f $train_list
fi
if [ $made_test_list ] && [ -f $test_list ]; then
    rm -f $test_list
fi

