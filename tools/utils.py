import torch
from mmcv.parallel import collate, scatter
from mmedit.apis import init_model
from mmedit.datasets.pipelines import Compose
import multiprocessing
import threading
import os
import numpy as np
import pickle
import tifffile
import sys
from ctypes import *
import math
import torch
from torchvision.utils import make_grid
import copy

#svs labels
svs_desc='Aperio Image Library Fake\nABC |AppMag={mag}|Filename={filename}|MPP={mpp}'
label_desc='Aperio Image Library Fake\nlabel {W}x{H}'
macro_desc='Aperio Image Library\nmacro {W}x{H}'

mpp=0.25   #40 times
mag=40
resolution=[10000/mpp,10000/mpp, 'CENTIMETER']

def get_tissue_regions(path_mask):
    w,h=path_mask.shape
    xmin=-1  #from index=0
    xmax=-1
    ymin=-1
    ymax=-1

    for i in range(w):
        count=sum(path_mask[i,:])
        if count>0:
            xmin=i
            xmax=i
            break

    for i in range(w-1,-1,-1):
        count=sum(path_mask[i,:])
        if count>0:
            xmax=i
            break

    for i in range(h):
        count=sum(path_mask[:,i])
        if count>0:
            ymin=i
            ymax=i
            break

    for i in range(h-1,-1,-1):
        count=sum(path_mask[:,i])
        if count>0:
            ymax=i
            break

    return [xmin,xmax,ymin,ymax]

def gen_im(img,factor):
    imgarray = img[0:-1:factor, 0:-1:factor, :]
    return  imgarray

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[0, 1, 2], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[0, 1, 2], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def savedata(wsidata,outputname, minfactor=3,tilesize=256):
    print('beging reading data')
    thumbmail_im = wsidata[0:-1:32, 0:-1:32, :]
    compression=['JPEG',95,dict(outcolorspace='YCbCr')]
    kwargs=dict(subifds=0,photometric='rgb',planarconfig='CONTIG',compression=compression,dtype=np.uint8,metadata=None)
    filename=outputname.split('/')[-1]
    with tifffile.TiffWriter(outputname,bigtiff=True,ome=True) as tif:
        for i in range(minfactor):
            print('beging svs at %d'%(i))
            factor=2**i
            gen = gen_im(wsidata,factor)

            if i==0:
                desc=svs_desc.format(mag=mag,filename=filename,mpp=mpp)
                tif.write(data=gen,tile=(tilesize,tilesize),description=desc, **kwargs)
                tif.write(data=thumbmail_im,description='',**kwargs)
            else:
                tif.write(data=gen, tile=(tilesize,tilesize), description='',**kwargs)

        tif.write(data=thumbmail_im,subfiletype=1,
                  description=label_desc.format(W=thumbmail_im.shape[1],H=thumbmail_im.shape[0]),**kwargs)
        tif.write(data=thumbmail_im,subfiletype=9,
                  description=label_desc.format(W=thumbmail_im.shape[1],H=thumbmail_im.shape[0]),**kwargs)


def read_mask(pathname,infoext):
    basename=pathname.split('/')[-1]
    infoname=os.path.join(pathname,basename+infoext)
    fwd=open(infoname,'rb')
    split_height=pickle.load(fwd)
    split_width=pickle.load(fwd)
    sampleheight=pickle.load(fwd)
    samplewidth=pickle.load(fwd)
    path_mask=pickle.load(fwd)
    return path_mask

def preprocess_output(output_imagepath,pathsize,infoext='_info'):
    path_mask=read_mask(output_imagepath,infoext)
    [xmin, xmax, ymin, ymax] = get_tissue_regions(path_mask)
    wsidata = np.ones((pathsize * (xmax - xmin + 1), pathsize * (ymax - ymin + 1), 3),dtype=np.uint8)*255
    return xmin,  ymin, wsidata

def postprocess_output(wsidata,imgname,data,xmin, ymin,patchsize):
    for onename, one in zip(imgname, data):
        parts = os.path.basename(onename).split('_')
        i, j = int(parts[-3]), int(parts[-2])
        wsidata[(i - xmin) * patchsize:(i - xmin + 1) * patchsize, (j - ymin) * patchsize:(j - ymin + 1) * patchsize,
        :] = tensor2img(one)

def init_all(config, checkpoint, device):
    model = init_model(config, checkpoint, device=device)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    return model,test_pipeline

def inference(model, imgs,data,wsidata, xmin,  ymin,patchsize,batchsize):
    count = len(imgs) // batchsize

    for i in range(count+1):
        tempdata={'meta':data['meta'],'lq':data['lq'][i*batchsize:(i+1)*batchsize]}
        if len(tempdata['lq'])==0:
            continue

        tempimgs=imgs[i*batchsize:(i+1)*batchsize]
        # forward the model
        with torch.no_grad():
            result = model(test_mode=True, **tempdata)

        postprocess_output(wsidata,tempimgs,result['output'],xmin,ymin,patchsize)
        sys.stdout.write('\r>> generate image %d/%d ' % (i, count))
        sys.stdout.flush()

    print("finished inferecne")

def load_imgs(selected_sets,test_pipeline,results,i):
    # prepare data
    data = []
    for img in selected_sets:
        tempdata = dict(lq_path=img)
        tempdata = test_pipeline(tempdata)
        data.append(tempdata)

    results[i] = {'imgs':selected_sets,'data':data}

def loaddata(imgs,test_pipeline,batchsize,device=0,workers=1):
    process = []
    results = multiprocessing.Manager().dict()

    for i in range(workers):
        selected_sets = imgs[i:len(imgs):workers]  # the i workers' job
        results[str(i)] = 0
        p = multiprocessing.Process(target=load_imgs,args=(selected_sets,test_pipeline,results,str(i)))
        process.append(p)

    for i in range(workers):
        process[i].start()

    for i in range(workers):
        process[i].join()

    out_imgs=[]
    out_data=[]
    for i in range(workers):
        out_imgs.extend(results[str(i)]['imgs'])
        out_data.extend(results[str(i)]['data'])

    out_data = collate(out_data, samples_per_gpu=batchsize)
    out_data = scatter(out_data, [device])[0]

    for i in range(workers):
        process[i].terminate()

    return out_imgs, out_data

