import argparse
import tqdm
import os

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.optim import Adam

from dataloader import get_loader
from model import get_deeplab_resnet_101, get_deeplab_resnet_50


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder', type=str, default='')
    parser.add_argument('--anno_folder', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    
    args = parser.parse_args()
    return args


def train_model(model, device, loader, criterion, optimizer, epochs,
                save_folder):

    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        print (epoch)

        for imgs, masks in loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            out = model(imgs)['out']

            loss = criterion(out, masks)
            print (loss)
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        print ('saving to :', save_folder)

        save_path = os.path.join(save_folder, 'deeplab_' + str(epoch) + '.ckpt')
        torch.save(model.state_dict(), save_path)
        
    
if __name__ == '__main__':

    args = get_args()

    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_deeplab_resnet_50(20)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_loader(image_folder=args.image_folder, anno_folder=args.anno_folder,
                              num_batches=args.batch_size)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, device, train_loader, criterion, optimizer, args.epoch_num,
                args.save_folder)
    
    
