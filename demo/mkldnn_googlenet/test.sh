set -e
cfg=googlenetv1.py
batch_size=64
use_dummy=1
use_mkldnn=1

./run.sh test $cfg $batch_size $use_dummy $use_mkldnn
