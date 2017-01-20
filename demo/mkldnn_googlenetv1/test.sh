set -e
output=./models_googlenet
model=models_googlenet/pass-00019/
function train() {
  cfg=$1
  thread=$2
  bz=$3
  args="batch_size=$3"
  prefix=$4
  task="test"
  log="log_${task}_$prefix-${thread}cpu-$bz.log"
  rm -f $log
  paddle train --job=$task \
    --config=$cfg \
    --use_gpu=False \
    --trainer_count=$thread \
    --log_period=1 \
    --init_model_path=$model \
    --test_period=100 \
    --config_args=$args \
    2>&1 | tee -a $log 2>&1 
}

#========single-gpu=========#
# alexnet
#train alexnet.py 1 64 alexnet
#train alexnet.py 1 128 alexnet
#train alexnet.py 1 256 alexnet
#train alexnet.py 1 512 alexnet
#

# googlenet
train googlenet.py 1 64 googlenet
#train googlenet.py 1 128 googlenet
#train googlenet.py 1 256 googlenet

# smallnet
#train smallnet_mnist_cifar.py 1 64 smallnet
#train smallnet_mnist_cifar.py 1 128 smallnet
#train smallnet_mnist_cifar.py 1 256 smallnet
#train smallnet_mnist_cifar.py 1 512 smallnet


############################
#========multi-gpus=========#
#train alexnet.py 4 512 alexnet
#train alexnet.py 4 1024 alexnet

#train googlenet.py 4 512 googlenet 
#train googlenet.py 4 1024 googlenet
