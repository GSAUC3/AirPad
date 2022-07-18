import torch 
from torch import nn 
from PIL import Image as im 
import numpy as np

class Cnet(nn.Module):
    def __init__(self) -> None:
        super(Cnet,self).__init__()
        self.convlayers = nn.Sequential(
                nn.Conv2d(3,8,(5,5),(1,1),padding=0),
                nn.ReLU(), # max(0,x)
                nn.MaxPool2d(2,2,0),#12

                nn.Conv2d(8,16,3,1,0),#10
                nn.ReLU(),
                nn.MaxPool2d(2,2,0),#5

                nn.Conv2d(16,32,3,1,0),
                nn.ReLU()      #4          
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9*32,out_features=256,bias=True),
            nn.ReLU(),
            nn.Linear(256,66,bias=True)
        )
    
    def forward(self,x):
        x = self.convlayers(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        # x = self.classifier(x.view(-1,16*32))
        return x

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# image = im.open('applePie.jpg')
image = im.open('pach.jpg')
imagearray = np.asarray(image)

x,y = 0,160
crop = imagearray[x:x+160,y:y+160]

crop= im.fromarray(crop)
c =crop.resize((28,28), im.ANTIALIAS)


model = Cnet()
model.load_state_dict(torch.load('./trainedmodel/model.pth'))

def predict(image_array):
    model.eval()
    
    datapoint = torch.from_numpy(np.asarray(image_array))
    datapoint = datapoint.permute(2,0,1)
    datapoint = datapoint.unsqueeze(0)
    out = model(datapoint.float())
    out = nn.functional.softmax(out,1)
    
    print(classes[out.argmax(1).item()])

predict(c)