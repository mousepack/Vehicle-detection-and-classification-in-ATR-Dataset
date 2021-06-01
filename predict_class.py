from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms





class classifier():
    def __init__(self,model_path):
        self.model = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['2S3','BMP2','BRDM2','BTR70','D20_MTLB','PICKUP','SUV','T72','ZSU23']
        self.transforms = transforms.Compose([
                    
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                                                                ])

    def predict(self,image_raw):
        image = Image.fromarray(image_raw.astype('uint8'), 'RGB')
        image_tensor = self.transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        output = self.model(image_tensor)
        i = output.argmax().item()
        return self.class_names[i]
   

