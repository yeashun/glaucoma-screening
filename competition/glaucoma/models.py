import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self,name,pretrained=False):
        super().__init__()
        if name=='efficientnet_v2_s':
            net=models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=net.features
            self.dim=net.classifier[1].in_features
            self.norm=nn.Identity()
        elif name=='efficientnet_b0':
            net=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=net.features
            self.dim=net.classifier[1].in_features
            self.norm=nn.Identity()
        elif name=='convnext_tiny':
            net=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=net.features
            self.dim=net.classifier[2].in_features
            self.norm=net.classifier[0]
        else:
            raise ValueError(name)

    def forward(self,x,return_map=False):
        fmap=self.features(x)
        pooled=self.norm(fmap.mean(dim=(-2,-1),keepdim=True)).flatten(1)
        return (pooled,fmap) if return_map else pooled

class FundusClassifier(nn.Module):
    def __init__(self,config,pretrained=False):
        super().__init__()
        self.config=config
        self.encoder=Encoder(config.get('backbone','efficientnet_v2_s'),pretrained)
        self.use_age=config.get('use_age',True)
        self.age_mlp=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU()) if self.use_age else None
        self.kind=config.get('kind','global')
        self.local_encoder=Encoder('efficientnet_b0',pretrained) if self.kind in ('local','fusion') else None
        dim=(self.encoder.dim if self.kind!='local' else 0)+(self.local_encoder.dim if self.local_encoder else 0)+(16 if self.use_age else 0)
        self.head=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,1))
        self.freeze_global=config.get('freeze_global',False)
        self.freeze_local=config.get('freeze_local',False)

    def forward(self,image,age,local=None):
        features=[]
        if self.kind!='local':
            if self.freeze_global:
                with torch.no_grad():
                    features.append(self.encoder(image))
            else:
                features.append(self.encoder(image))
        if self.local_encoder is not None:
            if self.freeze_local:
                with torch.no_grad():
                    features.append(self.local_encoder(local))
            else:
                features.append(self.local_encoder(local))
        if self.use_age:
            features.append(self.age_mlp(age))
        return self.head(torch.cat(features,dim=1)).flatten()

    def classify_features(self,global_features,age,local_features=None):
        features=[]
        if self.kind!='local':
            features.append(global_features)
        if local_features is not None:
            features.append(local_features)
        if self.use_age:
            features.append(self.age_mlp(age))
        return self.head(torch.cat(features,dim=1)).flatten()

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_global:
            self.encoder.eval()
        if self.freeze_local and self.local_encoder is not None:
            self.local_encoder.eval()
        return self

def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=FundusClassifier(checkpoint['config'],pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model,checkpoint
