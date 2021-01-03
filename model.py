import argparse
import torch
import numpy as np

import torchvision

from torchvision import models
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead



def get_deeplab_resnet_101(num_class, pretrained=True):

    model = models.segmentation.deeplabv3_resnet101(
        pretrained=pretrained, progress=True)

    model.classifier = DeepLabHead(2048, num_class)

    return model


def get_deeplab_resnet_50(num_class, pretrained=True):

    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,
                                                   progress=True)
    model.classifier = DeepLabHead(2048, num_class)

    return model

#for this I wanna implement losses


if __name__ == '__main__':

    model = get_deeplab_resnet_101(num_class=19)
    #model = get_deeplab_resnet_50(19)
    
    model.train()

    torch.save(model.state_dict(), 'model.ckpt')
    
    input_tensor = torch.randn(2, 3, 128, 128)

    out = model(input_tensor)['out']

    print (out.shape)
    
    


