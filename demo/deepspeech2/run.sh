#!/bin/bash
set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

function usage() {
    echo "run.sh task (batch_size) (use_dummy) (use_mkldnn) (use_mkldnn_wgt) (layer_num)"
    echo "One required inputs: "
    echo "    task    : train/test/pretrain/time."
    echo "The inputs with brackets are optional."
}

if [ $# -lt 1 ]; then
    echo "At least one input!"
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
task=$1
if [ $task != "train" ] && [ $task != "time" ] \
 && [ $task != "test" ] && [ $task != "pretrain" ]; then
    echo "unknown task: ${task}"
    usage
    exit 0
fi

### other inputs ###
topology="deepspeech2"
config="ds2.py"
layer_num=6
bs=2
use_dummy=1
use_mkldnn=1
use_mkldnn_wgt=1
models_in=
models_out=
log_dir=
is_test=0
if [ $task == "test" ]; then
    is_test=1
    use_mkldnn_wgt=0 # suppose the models are trained by CPU, so compatible with CPU weight
fi

# default use mkldnn and mkldnn_wgt, and do not use dummy data when training
if [ $task == "train" ] || [ $task == "pretrain" ]; then
    use_dummy=1
    use_mkldnn=1
    use_mkldnn_wgt=1
fi
if [ $2 ]; then
    bs=$2
fi
if [ $3 ]; then
    use_dummy=$3
fi
if [ $4 ]; then
    use_mkldnn=$4
fi
if [ $5 ]; then
    use_mkldnn_wgt=$5
fi

models_in=models/${topology}/pass-00001/
models_out=./models/${topology}
log_dir=logs/${topology}

# prepare log
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi
if [ $use_dummy -eq 1 ]; then
    log="${log_dir}/log_${task}_${topology}_bs${bs}_dummy.log"
else
    log="${log_dir}/log_${task}_${topology}_bs${bs}.log"
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
use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test},layer_num=${layer_num}"

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
    --trainer_count=1 \
    --rnn_use_batch=False \
    --feed_data=False \
    --log_period=1 \
    --test_period=10 \
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

