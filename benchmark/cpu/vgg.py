#!/usr/bin/env python
from paddle.trainer_config_helpers import *

batch_size = get_config_arg('batch_size', int, 64)
layer_num = get_config_arg('layer_num', int, 16)
is_predict = get_config_arg("is_predict", bool, False)
is_test = get_config_arg("is_test", bool, False)
use_dummy = get_config_arg("use_dummy", bool, False)
data_provider = get_config_arg("data_provider", bool, True)
####################Data Configuration ##################
img_size = 256
crop_size = 224
data_size = 3 * crop_size * crop_size
num_classes = 1000
label_size = 1
if not is_predict and data_provider:
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
img = data_layer(name='image', size=data_size)
def vgg_network(vgg_num=3):
    tmp = img_conv_group(
        input=img,
        num_channels=3,
        conv_padding=1,
        conv_num_filter=[64, 64],
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_size=2,
        pool_stride=2,
        pool_type=MaxPooling())

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[128, 128],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    channels = []
    for i in range(vgg_num):
        channels.append(256)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    channels = []
    for i in range(vgg_num):
        channels.append(512)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=channels,
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    tmp = fc_layer(
        input=tmp,
        size=4096,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    return fc_layer(input=tmp, size=num_classes, act=SoftmaxActivation())

if layer_num == 16:
    output = vgg_network(3)
elif layer_num == 19:
    output = vgg_network(4)
else:
    print("Wrong layer number.")

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    inputs(img, lbl)
    outputs(classification_cost(name='loss', input=output, label=lbl))
else:
    outputs(output)
