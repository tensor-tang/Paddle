set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

output=./models_googlenet
model=models_googlenetv1/pass-00019/
train_list="data/train.list"
test_list="data/test.list"
if [ ! -d "data" ]; then
    mkdir -p data
fi

function run() {
    task=$1
    cfg=$2
    bs=$3
    use_dummy=$4
    prefix="googlenet"
    is_test=0
    if [ $task == "test" ]; then
        is_test=1
    fi
    log="log_${task}_${prefix}_bs${bs}.log"
    rm -f $log
    if [ ! -f $train_list ] && [ $task == "train" ]; then
        if [ $use_dummy -eq 1 ]; then
            echo " " > $train_list
        else
            echo "$train_list does not exist!"
            exit 0
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
    if [ $is_test -eq 1 ] ; then
        if [ ! -d $model ]; then
          echo "model does not exist!"
        fi
    fi
    args="use_mkldnn=1,batch_size=${bs},is_test=${is_test},use_dummy=${use_dummy}"
    paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --log_period=1 \
    --init_model_path=$model \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
}

# googlenet
run test googlenetv1.py 64 1
#run train googlenetv1.py 64 0
