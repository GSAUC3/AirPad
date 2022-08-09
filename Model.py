import torch 
from torch import nn 
from PIL import Image as im 
import numpy as np
from torchvision import models

# https://github.com/GSAUC3/

class Cnn(nn.Module):
    def __init__(self) -> None:
        super(Cnn,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,64,5,1), 
            nn.ReLU(), # 24x 24
            nn.MaxPool2d(2,2), # 16x 12 x 12

            nn.Conv2d(64,128,3), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32x5x5

            nn.Conv2d(128,128,5), # 64x 1x14
            nn.ReLU(),

            nn.Conv2d(128,36,1,1) # 10x1x1
            
        )
    def forward(self,x):
        x = self.conv(x)
        return x

model =  Cnn()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


model.load_state_dict(torch.load('trainedmodel/model.pth'))

def predict(image_array):
    model.eval()
    
    crop= im.fromarray(image_array)
    crop =crop.resize((28,28), im.ANTIALIAS)
    datapoint = torch.from_numpy(np.asarray(crop))
    datapoint = datapoint.permute(2,0,1)
    datapoint = datapoint.unsqueeze(0)
    out = model(datapoint.float())
    out = nn.functional.softmax(out,1)
    
    return classes[out.argmax(1).item()]


def sliding_window(image):
    imgStack =[]
    letters = []

    x,y = 0,0
    height,width,_= image.shape
    stride = width//4
    # print(f'height {height} width {width} stride {stride}')
    for x in range(2):
        for y in range(4):
            imgStack.append(image[x*stride:(x+1)*stride,y*stride:(y+1)*stride])

    for i in imgStack:  
        # print(i.shape)
        if np.sum(i)>10:
            pred = predict(i)
            letters.append(pred)

    return letters


