import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from mydataset import MyDataset
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_val_test_split(train_size, val_size, test_size,
                          train_transform, val_test_transform, batch_size, labels) -> list[DataLoader]:
    val_sample = np.random.choice(np.arange(train_size), size=(val_size,), replace=False)
    val_indicator = np.zeros(train_size, dtype=bool)
    val_indicator[val_sample] = True
    train_indicator = ~val_indicator
    train_ind = np.arange(train_size)[train_indicator]
    val_ind = np.arange(train_size)[val_indicator]

    trainset = MyDataset(train_ind, False, transform=train_transform, labels=labels)
    valset = MyDataset(val_ind, False, transform=val_test_transform, labels=labels)
    testset = MyDataset(np.arange(test_size), True, transform=val_test_transform, labels=labels)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def save_results(model, test_loader, test_size, filename) -> None:
    test_labels = torch.Tensor(0).to(device)
    for data, _ in tqdm(test_loader):
        data = data.to(device)

        with torch.no_grad():
          logits = model(data)

        pred = logits.argmax(dim=1)
        test_labels = torch.cat((test_labels, pred))
        
    picture_ids = ["test_" + f"{str(i).zfill(5)}" + ".jpg" for i in np.arange(test_size)]
        
    ans = pd.DataFrame({"Id": picture_ids,
                        "Category": test_labels.cpu()})
    ans.to_csv(f"/kaggle/working/{filename}.csv", index=False)


def save_results_with_augs(model, test_loader, test_size, filename) -> None:
    test_labels = torch.Tensor(0).to(device)
    for data, _ in tqdm(test_loader):
        data = data.to(device)

        data_verflip = v2.RandomVerticalFlip(p=1)(data)
        data_horflip = v2.RandomHorizontalFlip(p=1)(data)

        with torch.no_grad():
          logits1 = model(data)
          logits2 = model(data_verflip)
          logits3 = model(data_horflip)
          logits = (logits1 + logits2 + logits3) / 3

        pred = logits.argmax(dim=1)
        test_labels = torch.cat((test_labels, pred))
        
    picture_ids = ["test_" + f"{str(i).zfill(5)}" + ".jpg" for i in np.arange(test_size)]
        
    ans = pd.DataFrame({"Id": picture_ids,
                        "Category": test_labels.cpu()})
    ans.to_csv(f"/kaggle/working/{filename}.csv", index=False)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
