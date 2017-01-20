set -e
output=./models_googlenet
model=models_googlenetv1/pass-00019/
function train() {
  cfg=$1
  bs=$2
  args="batch_size=$2"
  prefix=$3
  task="test"
  log="log_${task}_${prefix}_bs${bs}.log"
  rm -f $log
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
train googlenetv1.py 64 googlenetv1

