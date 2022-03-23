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


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    model = ResNet(block=BasicBlock, layers=[2,2,2,2], num_classes=10).to(device=device)
    model.model_name = "resnet18"
    
    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")
    
    onnx_dir = args.onnx_dir+model.model_name+".onnx"
    dummp_input = torch.randn(args.batch_size, 3, 224, 224).to(device)
    onnx.export(model, dummp_input, onnx_dir, export_params=True)
    # torch.onnx.utils.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_dir)), onnx_dir)

    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    writer = SummaryWriter(args.log_dir)
    
    print("start train...\n")
    for epoch in range(1, args.num_epochs+1):
        print("Epoch: {:<5d}\n".format(epoch))
        
        # train ----------------------------------------------------
        model.train()
        running_loss, corrects, total = 0, 0, 0

        start_time = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            corrects += predicted.eq(labels).sum().item()
            
            # 记录某一个变量随训练过程的变化
            writer.add_scalar('train_loss', loss.item(), global_step=batch_idx+epoch*(len(train_loader)+1))
            
            if batch_idx % args.print_itr == 0 or batch_idx==len(train_loader)-1:
                print("batch_idx:{:<5d}, len(train_loader):{:<5d}, Loss:{:.3f} | ACC:{:.3f}% ".format(
                    batch_idx, len(train_loader), running_loss/(batch_idx+1), 100.*corrects/total
                ))
                
            loss.backward()
            optimizer.step()
            
        end_time = time.time()
        print("Train time Usage:{:.2f}".format(end_time-start_time))
            
        train_loss.append(running_loss/len(train_loader))
        train_acc.append(corrects/total)
            
        # val -----------------------------------------------------------
        model.eval()
        running_loss, corrects, total = 0, 0, 0
        print("start val---------------------------------------------")
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                corrects += predicted.eq(labels).sum().item()
                
                if batch_idx % args.print_itr == 0 or batch_idx==len(val_loader)-1:
                    # format数字格式化：https://blog.csdn.net/menghuanshen/article/details/104258081
                    print("batch_idx:{:<5d}, len(val_loader):{:<5d}, Loss:{:.3f} | ACC:{:.3f}%% ".format(
                        batch_idx, len(val_loader), running_loss/(batch_idx+1), 100.*corrects/total
                    ))
        val_loss.append(running_loss/len(val_loader))
        val_acc.append(corrects/total)          
        print("val over---------------------------------------------\n")
        
        scheduler.step()
        
        
        if epoch % args.save_epoch == 0 or epoch == args.num_epochs+1:
            state = {
                "name": model.model_name,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "criterion": criterion.state_dict(),
                "scheduler": scheduler.state_dict(),
                "acc": 100.*corrects/total,
                "epoch": epoch
            }
            file_name = time.strftime(model.model_name + " epoch" + str(epoch) + "  %m/%d %H:%M:%S.pth")
            path = os.path.join(args.model_save_path, file_name)
            torch.save(state, path)
        
            plot_metrics(series=[train_acc, val_acc], labels=['Train', 'Val'], 
                        xlabel='Epoch', ylabel='Acc', 
                        xticks=np.arange(0, args.num_epochs+1, args.num_epochs//10),
                        yticks=np.arange(0, 1, 0.2),
                        save_path=args.fig_dir + model.model_name + "_epoch" + str(epoch) + "_Acc.png")
            plot_metrics(series=[train_loss, val_loss], labels=['Train', 'Val'], 
                        xlabel='Epoch', ylabel='Loss', 
                        xticks=np.arange(0, args.num_epochs+1, args.num_epochs//10),
                        yticks=None,
                        save_path=args.fig_dir + model.model_name + "_epoch" + str(epoch) + "_Loss.png")