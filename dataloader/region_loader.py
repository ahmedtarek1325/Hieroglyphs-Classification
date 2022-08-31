from torchvision import transforms 
from torchvision.transforms import InterpolationMode
import torch
import numpy as np

def transfomer(): 
    data_transform= transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,)),
                                        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)])
    return data_transform


def region_generator(img,regions,batchsize,transfomerfunc= transfomer):
    '''
    INPUT 
    - img: oringinal image that we want to etract certain regions from it 
    - regions: Tuple that has values represented as (y1,y2,x1,x2)
    - batchsize: how many regions u want us to etract at one iteration 
    - transforms: the required transforms to be done on regions before being fed to the classifier 
    OUTPUT
    - list of images 
    - regions arrayof tuples
    '''
    transform= transfomerfunc()

    imglist= np.array([])
    newregions= []
    for i,v in enumerate(regions):
        newimg=transform(img[v[0]:v[1],v[2]:v[3]]).unsqueeze(0)
        imglist= np.append(imglist,newimg,axis=0) if len(imglist.shape)>1 else newimg
        newregions.append(v)
        #print("shape of image list",imglist.shape)
       
        #print("imglist ",imglist.shape)
        #imglist= torch.cat((imglist,newimg),axis=0) if len(imglist.shape)>1 else newimg
        
        if i%batchsize == 0: 
            yield torch.Tensor(imglist),newregions
            imglist= np.array([])
            newregions= [] 

    yield torch.Tensor(imglist),newregions

