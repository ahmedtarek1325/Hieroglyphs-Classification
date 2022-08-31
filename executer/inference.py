import torch
import numpy as np 


def Inference(model,data_gen):
    '''
    INPUT 
    - Classifier model 
    - data_gen: image generator that returns (imgs,thier regions)
    OUTPUT 
    - Predictionss of the images 
    - regions of the images that are subset from the original image
    
    '''
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    predictions= [] 
    region_list=[] 
    with torch.no_grad():
        for i,(data,regions) in enumerate(data_gen): 

            data= data.to(device)
            
            outputs = model(data)
            outputs = outputs.detach().cpu().numpy()
            
            predictions.extend(np.argmax(outputs, axis=1))
            region_list.extend(regions)
    
    return predictions,region_list
