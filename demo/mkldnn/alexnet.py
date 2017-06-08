#!/usr/bin/env python
from paddle.trainer_config_helpers import *

batch_size = get_config_arg('batch_size', int, 64)
is_test = get_config_arg("is_test", bool, False)
use_dummy = get_config_arg("use_dummy", bool, False)
####################Data Configuration ##################
img_size = 256
crop_size = 227
data_size = 3 * crop_size * crop_size
num_classes = 1000
label_size = num_classes
train_list = 'data/train.list' if not is_test else None
test_list = 'data/test.list'
args = {
    # mean_path or mean_value only choose one
    'mean_value': [103.939, 116.779, 123.68],
    'img_size': img_size,
    'crop_size': crop_size,
    'num_classes': num_classes,
    'use_jpeg': True,
    'color': True
}

define_py_data_sources2(
    train_list,
    test_list,
    module='dummy_provider' if use_dummy else 'image_provider',
    obj='processData',
    args=args)

######################Algorithm Configuration #############
settings(
    batch_size=batch_size,
    learning_rate=0.001 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))

#######################Network Configuration #############
# conv1
net = data_layer('data', size=data_size)
net = img_conv_layer(
    input=net,
    filter_size=11,
    num_channels=3,
    num_filters=96,
    stride=4)
net = img_lrn_layer(input=net, local_size=5, scale=0.0001, power=0.75)
net = img_pool_layer(input=net, pool_size=3, stride=2)

# conv2
net = img_conv_layer(
    input=net, filter_size=5, num_filters=256, stride=1, padding=2, groups=2)
net = img_lrn_layer(input=net, local_size=5, scale=0.0001, power=0.75)
net = img_pool_layer(input=net, pool_size=3, stride=2)

# conv3
net = img_conv_layer(
    input=net, filter_size=3, num_filters=384, stride=1, padding=1, groups=1)
# conv4
net = img_conv_layer(
    input=net, filter_size=3, num_filters=384, stride=1, padding=1, groups=2)

# conv5
net = img_conv_layer(
    input=net, filter_size=3, num_filters=256, stride=1, padding=1, groups=2)
net = img_pool_layer(input=net, pool_size=3, stride=2)

net = fc_layer(
    input=net,
    size=4096,
    act=ReluActivation(),
    layer_attr=ExtraAttr(drop_rate=0.5))
net = fc_layer(
    input=net,
    size=4096,
    act=ReluActivation(),
    layer_attr=ExtraAttr(drop_rate=0.5))
net = fc_layer(input=net, size=num_classes, act=SoftmaxActivation())

lab = data_layer('label', label_size)
loss = cross_entropy(input=net, label=lab)
outputs(loss)
