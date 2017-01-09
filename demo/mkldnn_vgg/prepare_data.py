import io
import sys,os
import numpy as np
from PIL import Image

my_dir = '/home/tangjian/imagedata_val/'
file_list = 'data/val.txt'
tmp = 'data/tmp.txt'
total_imgs = 0
with open(tmp, 'w') as ftrain:
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

train_list = 'data/train.list'
train_part_prefix = 'data/train_list.part'
steps = 1000  # every 1000 images save in one part

with open(train_list, 'w') as ftrain:
    with open(tmp, 'r') as ftmp:
        lines = [line.strip() for line in ftmp]
        img_idx = 0
        part_idx = 1
        part_path = train_part_prefix + str(part_idx)
        print(part_path)
        fpart = open(part_path, 'w')
        ftrain.write(part_path + '\n')
        for file_name in lines:
            img_idx = img_idx + 1
            if img_idx > steps:
                img_idx = 0
                part_idx = part_idx + 1
                part_path = train_part_prefix + str(part_idx)
                print(part_path)
                fpart.close()
                fpart = open(part_path, 'w')
                ftrain.write(part_path + '\n')
            fpart.write(file_name + '\n')
        fpart.close()

            
            
            
            
            
            