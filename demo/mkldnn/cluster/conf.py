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

HOSTS = [
    "root@slavem",
    "root@slave13",
]

'''
workspace configuration
'''
# job dir for cluster
JOB_DIR = ".."
# logs dir for logs and models
LOG_DIR = "."

'''
network configuration
'''
#pserver nics
PADDLE_NIC = "eno3"
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 1
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 0

#trainer config 
BATCH_SIZE_PER_NODE = 64
# use MKLDNN layers
PADDLE_USE_MKLDNN = 1
# if do not use mkldnn_weight, will use the same weight as CPU layers
PADDLE_USE_MKLDNN_WGT = 1
# if do not have training data, can use dummy data to test benchmark
PADDLE_USE_DUMMY = 0


#environments setting for all processes in cluster job
LD_LIBRARY_PATH = "/usr/lib64:/usr/local/lib:/usr/local/cuda/lib64"
