import os
import glob
import numpy as np

lq_folder='/media/disk3/sr_256/val/5X'
gt_folder='/media/disk3/sr_256/val/40'
ann_file='/media/disk3/sr_256/val/anno_part_file.txt'
ext='jpg'
maxfile=200

files=glob.glob(os.path.join(lq_folder,'*.'+ext))
fw=open(ann_file,'w')
np.random.shuffle(files)

count=0
for file in files:
    basename=os.path.basename(file)
    line=basename+' '+basename
    fw.writelines(line+'\n')
    count+=1
    if count>maxfile:
        break

fw.close()

