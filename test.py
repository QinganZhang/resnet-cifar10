import time
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset.get_dataloader import get_dataloader
from models import BasicBlock, BottleNeck, ResNet
from utils import plot_metrics

from tensorboardX import SummaryWriter
from torch import onnx
import netron


def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    model = ResNet(block=BasicBlock, layers=[2,2,2,2], num_classes=10).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    checkpoint = torch.load(args.load_dir)
    model.load_state_dict(checkpoint['model'])
    model.model_name = checkpoint['name']
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['criterion'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    test_loader = get_dataloader(args, "test")
            
    model.eval()
    test_loss, corrects, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            corrects += predicted.eq(labels).sum().item()
                
    print("len(val_loader):{:<5d}, Loss:{:.3f} , ACC:{:.3f}%% ".format(
         len(test_loader), test_loss/(batch_idx+1), 100.*corrects/total
    ))
                
    scheduler.step()    
