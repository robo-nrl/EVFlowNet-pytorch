import os
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from config import configs, print_args
from data_loader import EventData
from EVFlowNet import EVFlowNet
from test import test
from util import flow2rgb, AverageMeter, save_checkpoint
import pkbar
import time

def train(args, TrainLoader, model, optimizer, loss_fun, epoch):   
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kbar = pkbar.Kbar(target=len(TrainLoader), width=10)

    model.train()
    end = time.time()

    for idx, (input, gray) in enumerate(TrainLoader):
        input = input.to(args.device)
        gray = gray.to(args.device)

        if torch.sum(input) > 0:
            # measure data loading time
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            flow_dict = model(input)

            loss = loss_fun(flow_dict, gray[:, 0], gray[:, 1], model)
            
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            kbar.update(idx, values=[('BPS', 1/batch_time.avg),
                                            ('Batch_Time', batch_time.avg), 
                                            ('Data_Time', data_time.avg),
                                            ('Loss', losses.avg),
                                            ])

    return losses.avg

def main():
    args = configs()
    best_EPE = -1
    print_args(args)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    TrainSet = EventData(data_folder_path=args.root_dir, split='train', dt=args.dt, transform=train_transform)
    TrainLoader = torch.utils.data.DataLoader(dataset=TrainSet, batch_size=args.batch_size, shuffle=True)
    
    TestSet = EventData(data_folder_path=args.root_dir, split='test', dt=args.dt, transform=test_transform)
    TestLoader = torch.utils.data.DataLoader(dataset=TestSet, batch_size=1, shuffle=False)


    # model
    model = EVFlowNet(args)
    if args.pretrained:
        model_data = torch.load(args.pretrained)
        model.load_state_dict(model_data['state_dict'])

    model = model.to(args.device)

    model = torch.nn.DataParallel(model).to(args.device)

    print('=> Everything will be saved to {}'.format(args.save_path))

    if args.evaluate:
        with torch.no_grad():
            best_EPE = test(args, TestLoader, model)
        return

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight)

    for epoch in range(args.start_epoch, args.epochs):
        print('\n*****************************************')
        print('Epoch: ' + str(epoch+1) + ' / ' + str(args.epochs))

        train_loss = train(args, TrainLoader, model, optimizer, loss_fun, epoch)

        if (epoch+1)%args.evaluate_interval == 0:
            # evaluate on validation set
            print('\n\nEvaluating ...')
            with torch.no_grad():
                EPE = test(args, TestLoader, model)

            if best_EPE < 0:
                best_EPE = EPE

            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            print('Best EPE: {}'.format(best_EPE))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
            }, is_best, args.save_path)

        if epoch % 4 == 3:
            scheduler.step()


    

if __name__ == "__main__":
    main()
