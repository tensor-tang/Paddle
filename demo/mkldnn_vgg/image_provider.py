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

import random
from paddle.utils.image_util import *
from paddle.trainer.PyDataProvider2 import *

def hook(settings, img_size, crop_size, num_classes, color, file_list, use_jpeg,
         is_train, **kwargs):
    settings.img_size = img_size
    settings.crop_size = crop_size
    settings.mean_img_size = settings.crop_size
    settings.num_classes = num_classes
    settings.color = color
    settings.is_train = is_train
    settings.use_jpeg = use_jpeg
    settings.file_list = file_list

    settings.is_swap_channel = kwargs.get('swap_channel', None)
    if settings.is_swap_channel is not None:
        settings.swap_channel = settings.is_swap_channel
        settings.is_swap_channel = True

    if settings.color:
        settings.img_input_size = settings.crop_size * settings.crop_size * 3
    else:
        settings.img_input_size = settings.crop_size * settings.crop_size

    settings.mean_path = kwargs.get('mean_path', None)
    settings.mean_value = kwargs.get('mean_value', None)
    # can not specify both mean_path and mean_value.
    assert not (settings.mean_path and settings.mean_value)
    if not settings.mean_path:
        settings.mean_value = kwargs.get('mean_value')
        sz = settings.crop_size * settings.crop_size
        settings.img_mean = np.zeros(sz * 3, dtype=np.single)
        for idx, value in enumerate(settings.mean_value):
            settings.img_mean[idx * sz:(idx + 1) * sz] = value
        settings.img_mean = settings.img_mean.reshape(3, settings.crop_size,
                                                      settings.crop_size)
    else:
        settings.img_mean = load_meta(settings.meta_path,
                                             settings.mean_img_size,
                                             src_size, settings.color)

    settings.input_types = [
        dense_vector(settings.img_input_size),  # image feature
        integer_value(settings.num_classes)  # labels
    ]
    settings.logger.info('Image short side: %s', settings.img_size)
    settings.logger.info('Crop size: %s', settings.crop_size)
    if settings.is_swap_channel:
        settings.logger.info('swap channel: %s', settings.swap_channel)
    settings.logger.info('DataProvider Initialization finished')

@provider(init_hook=hook, min_pool_size=1, pool_size=20)  # , should_shuffle=False) 
def processData(settings, file_list):
    """
    The main function for loading data.
    Load the batch, iterate all the images and labels in this batch.
    file_list: the batch file list.
    """
    print("-----------------", file_list)
    with open(file_list, 'r') as fpart:
        lines = [line.strip() for line in fpart]
        if settings.is_train:
            random.shuffle(lines)
        for file_name in lines:
            img_path, lab = file_name.strip().split(' ')
            img = Image.open(img_path)
        #    print("-----------------", img_path, int(lab.strip()))
            img.load()
            img = img.resize((settings.img_size, settings.img_size), Image.ANTIALIAS)
            img = np.array(img).astype(np.float32)
        #    print(len(img.shape), img.size)
            if len(img.shape) == 3:
        #        print("before------", img.shape[0], img.shape[1], img.shape[2])
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)
        #        print("after-------", img.shape[0], img.shape[1], img.shape[2])
            # swap channel
                if settings.is_swap_channel:
                    img = img[settings.swap_channel, :, :]
                img_feat = preprocess_img(
                         img, settings.img_mean, settings.crop_size,
                         settings.is_train, settings.color)
                yield img_feat.tolist(), int(lab.strip())
