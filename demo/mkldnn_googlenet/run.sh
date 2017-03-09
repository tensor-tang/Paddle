set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

prefix="googlenet"
output=./models_googlenet
model=models_googlenetv1/pass-00019/
train_list="data/train.list"
test_list="data/test.list"
if [ ! -d "data" ]; then
    mkdir -p data
fi

function usage() {
    echo "run test/train/pretrain (cfg) (bs) (use_dummy) (use_mkldnn)"
}
if [ $# -lt 1 ]; then
    echo "At least one input"
    usage
    exit 0
fi

## inputs:
task=$1
cfg=googlenetv1.py
bs=64
use_dummy=1
use_mkldnn=1
if [ $2 ]; then
    cfg=$2
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

## 
is_test=0
use_mkldnn_wgt=1
if [ $task == "test" ]; then
    is_test=1
    use_mkldnn_wgt=0
fi
log="log_${task}_${prefix}_bs${bs}.log"
rm -f $log
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
      echo "model does not exist!"
    fi
fi

args="batch_size=${bs},use_dummy=${use_dummy},use_mkldnn=${use_mkldnn},\
use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test}"

if [ $task == "train" ]; then
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --dot_period=1 \
    --log_period=2 \
    --test_period=100 \
    --trainer_count=1 \
    --num_passes=2 \
    --save_dir=$output \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
else  # pretrain or test
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --log_period=1 \
    --init_model_path=$model \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
fi

