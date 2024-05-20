import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from TrainTestLoops import train_with_wandb
from utils import train_val_test_split, save_results, set_random_seed
from v4ReceptiveResNets import resnet50


def experiment():
    set_random_seed(0)
    labels = pd.read_csv("labels.csv")

    kTrainSize = 100000
    kTestSize = 10000
    kValSize = 5000
    kBatchSize = 128

    train_transform = v2.Compose(
            [v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(3),]), p=0.2),
            v2.RandomApply(torch.nn.ModuleList([v2.RandomRotation((-15, 15)),]), p=0.2),
            v2.RandomApply(torch.nn.ModuleList([v2.ColorJitter(brightness=[0, 1], contrast=[0, 0.5]),]), p=0.2),
            v2.ToTensor(),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    val_test_transform = v2.Compose(
            [v2.ToTensor(),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = train_val_test_split(kTrainSize, kValSize, kTestSize,
                                                                train_transform, val_test_transform, kBatchSize, labels)
    
    n_epochs = 60
    n_classes = 200
    model = resnet50().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    train_with_wandb(model, optimizer, n_epochs, train_loader, val_loader, model_name='v4ReceptiveResNets',  scheduler=scheduler)
    
    save_results(model, test_loader, kTestSize, filename='test_labels')


if __name__ == "__main__":
    experiment()