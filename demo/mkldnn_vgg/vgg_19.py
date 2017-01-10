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

from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)
is_test = get_config_arg("is_test", bool, False)
data_provider = get_config_arg("data_provider", bool, True)
####################Data Configuration ##################
img_size = 256
crop_size = 224
data_size = 3 * crop_size * crop_size
num_classes = 1000
if not is_predict and data_provider:
    train_list = 'data/train.list' if not is_test else None
    test_list = 'data/train.list'
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
        module='image_provider',
        obj='processData',
        args=args)

######################Algorithm Configuration #############
batch_size = 64
settings(
    batch_size=batch_size,
    learning_rate=0.001 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))

#######################Network Configuration #############

img = data_layer(name='image', size=data_size)
def vgg_19_network(input_image, num_channels, num_classes=1000):
    """
    Same model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8

    :param num_classes:
    :param input_image:
    :type input_image: LayerOutput
    :param num_channels:
    :type num_channels: int
    :return:
    """

    tmp = img_conv_group(
        input=input_image,
        num_channels=num_channels,
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

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[256, 256, 256, 256],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)

    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[512, 512, 512, 512],
        conv_padding=1,
        conv_filter_size=3,
        conv_act=ReluActivation(),
        pool_stride=2,
        pool_type=MaxPooling(),
        pool_size=2)
    tmp = img_conv_group(
        input=tmp,
        conv_num_filter=[512, 512, 512, 512],
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

label_size = num_classes if not is_predict else 1
predict = vgg_19_network(input_image=img, num_channels=3, num_classes=num_classes)

# small_vgg is predefined in trainer_config_helpers.networks
#predict = small_vgg(input_image=img, num_channels=3, num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)
