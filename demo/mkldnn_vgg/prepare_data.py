import io
import sys,os
import numpy as np
from PIL import Image

my_dir = '/home/tangjian/imagedata_val/'
file_list = 'data/val.txt'
train_list = 'data/train.list'
train_list_actual = 'data/train.list.txt'
total_imgs = 0
with open(train_list_actual, 'w') as ftrain:
    with open(file_list, 'r') as fdata:
        lines = [line.strip() for line in fdata]
        for file_name in lines:
            abs_line = my_dir + file_name
            img_path, lab = abs_line.strip().split(' ')
            img = Image.open(img_path)
            img.load()
            img = np.array(img).astype(np.float32)
            if not len(img.shape) == 3:
                print(abs_line, "skip---------------------")
                continue
            ftrain.write(abs_line + '\n')
            total_imgs = 1 + total_imgs

print(total_imgs)

with open(train_list, 'w') as ftrain:
    ftrain.write(train_list_actual + '\n')

            
            
            
            
            
            