import os,glob
from utils import init_all,inference,loaddata,preprocess_output,savedata
import time
import threading
from time import sleep

input_path='/media/disk1/'
output_path='/media/disk3/'
config='../configs/restorers/Pathology_MSR/glean_in256out2048_pathology.py'
checkpoint='../pretrained/pretrained_MSR.pth'
device=0
batchsize=16
workers=8
patchsize=2048

model,test_pipeline=init_all(config,checkpoint,device)
dirs=os.listdir(input_path)
if os.path.exists(output_path):
    print('need a new path')
    #exit()
else:
    os.makedirs(output_path)

for dir in dirs:
    input_dir=os.path.join(input_path,dir+'/common_images')
    output_dir=os.path.join(output_path,dir+'/common_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dirs=os.listdir(input_dir)
    for image_dir in image_dirs:
        output_imagepath = os.path.join(output_dir, image_dir)
        if os.path.exists(output_imagepath):
            print(output_imagepath+' has existed and ignore')
            continue

        os.makedirs(output_imagepath)
        images=glob.glob(os.path.join(os.path.join(input_dir,image_dir),"*.jpg"))
        print('Processing the %s at %s'%(output_imagepath,time.asctime(time.localtime(time.time()))))

        source_file = os.path.join(os.path.join(input_dir, image_dir), image_dir + '_info')
        command = f'cp "{source_file}"  "{output_imagepath}"'
        os.system(command)

        print('the data loading')
        imgs,data=loaddata(images,test_pipeline,batchsize,device,workers)
        xmin,  ymin, wsidata=preprocess_output(output_imagepath, patchsize, infoext='_info')
        print('inference begining')
        inference(model, imgs, data, wsidata, xmin,  ymin,patchsize,batchsize)
        print('the data saving')
        one_basename = os.path.basename(output_imagepath)
        savedata(wsidata,os.path.join(output_imagepath, one_basename + '.svs'), minfactor=10,tilesize=256)
        print('End processing the %s at %s' % (output_imagepath, time.asctime(time.localtime(time.time()))))

