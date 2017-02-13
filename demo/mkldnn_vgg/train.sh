#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

config=vgg_19.py
output=./models_vgg_19
log=log_train.log

paddle train \
--config=$config \
--dot_period=1 \
--log_period=1 \
--test_all_data_in_one_period=0 \
--use_gpu=0 \
--trainer_count=1 \
--num_passes=2 \
--save_dir=$output \
--config_args="use_dummy=0,batch_size=32" \
2>&1 | tee $log

#--saving_period_by_batches=3 \  // not work

#python -m paddle.utils.plotcurve -i $log > plot.png
