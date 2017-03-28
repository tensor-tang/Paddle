#!/bin/sh

#python paddle.py \
#  --job_workspace="${PATH_TO_REMOTE_EXISTED_WORKSPACE}" \
#  --dot_period=10 \
#  --ports_num_for_sparse=2 \
#  --log_period=50 \
#  --num_passes=10 \
#  --trainer_count=4 \
#  --saving_period=1 \
#  --local=0 \
#  --config=./trainer_config.py \
#  --save_dir=./output \
#  --use_gpu=0

PADDLE_DIR=/home/tangjian/paddle-tj/
PATH_TO_LOCAL_WORKSPACE=${PADDLE_DIR}/benchmark/cluster/networks/
#cfg=vgg.py
cfg=googlenet.py
#cfg=resnet.py

python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --dot_period=1 \
  --ports_num_for_sparse=2 \
  --log_period=1 \
  --num_passes=10 \
  --trainer_count=1 \
  --saving_period=1 \
  --local=0 \
  --config=${cfg} \
  --save_dir=./output \
  --use_gpu=0
