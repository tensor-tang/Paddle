set -e
unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

output=./models_googlenet
model=models_googlenetv1/pass-00019/

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
