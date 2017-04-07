import io, os
import random
import numpy as np
from paddle.trainer.PyDataProvider2 import *


def initHook(settings, crop_size, num_classes, color, is_train, **kwargs):
    settings.crop_size = crop_size
    settings.color = color
    settings.num_classes = num_classes

    if is_train:
        settings.dummy_size = 256
    else:
        settings.dummy_size = 2048
    if settings.color:
        settings.data_size = settings.crop_size * settings.crop_size * 3
    else:
        settings.data_size = settings.crop_size * settings.crop_size

    settings.slots = [dense_vector(settings.data_size), integer_value(1)]


@provider(
    init_hook=initHook, min_pool_size=-1, cache=CacheType.CACHE_PASS_IN_MEM)
def processData(settings, file_list):
    for i in xrange(settings.dummy_size):
        img = np.random.rand(1, settings.data_size).reshape(-1, 1).flatten()
        lab = random.randint(0, settings.num_classes-1)
        yield img.astype('float32'), int(lab)
