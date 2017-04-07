import io
import sys,os
import numpy as np
from PIL import Image

my_dir = 'data/train/'  # 'data/test/'
input_list = 'data/train/train.txt'  # 'data/test/val.txt'
intermediate_list = 'data/train.list.txt'  # 'data/test.list.txt'
output_list = 'data/train.list'  # 'data/test.list'
total_imgs = 0
with open(intermediate_list, 'w') as ftrain:
    with open(input_list, 'r') as fdata:
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

with open(output_list, 'w') as ftrain:
    ftrain.write(intermediate_list + '\n')

            
            
            
            
            
            