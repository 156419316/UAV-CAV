import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import cv2 
import numpy as np
import os
from PIL import Image

import utils

if __name__ == '__main__':
    image_root = './BEV/'
    Encoder = utils.FeatureEncoder(3,1)
    
    for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (640, 640))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize the image
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            image = image / 255.0
            
            # Add batch dimension
            image = torch.unsqueeze(image, 0) 
            print(Encoder(image).shape)
            
            
           
    
    
   