import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
from IPython.display import clear_output
from torchvision.transforms import v2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in tqdm(loader):
        data = data.to(device)
        target = target.flatten().long().to(device)

        with torch.no_grad():
          logits = model(data)
          loss = nn.functional.cross_entropy(logits, target)

        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)

def train_epoch(model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.flatten().to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        acc_log.append(acc.item())

    return loss_log, acc_log

def train(model, optimizer, n_epochs, train_loader, val_loader, model_name, scheduler=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        val_loss, val_acc = test(model, val_loader)

        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        torch.save(model.state_dict(), f"/kaggle/working/epoch_{epoch}_model_{model_name}.pt")
        torch.save(optimizer.state_dict(), f"/kaggle/working/epoch_{epoch}optimizer_{model_name}.pt")
        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def test_with_wandb(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in tqdm(loader):
        data = data.to(device)
        target = target.flatten().long().to(device)

        with torch.no_grad():
          logits = model(data)
          loss = nn.functional.cross_entropy(logits, target)

        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        loss_log.append(loss.item())
        acc_log.append(acc.item())

    wandb.log({"test_acc": np.mean(acc_log), "test_loss": np.mean(loss_log)})

def test_with_wandb_and_augs(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in tqdm(loader):
        data = data.to(device)
        target = target.flatten().long().to(device)

        data_verflip = v2.RandomVerticalFlip(p=1)(data)
        data_horflip = v2.RandomHorizontalFlip(p=1)(data)

        with torch.no_grad():
          logits1 = model(data)
          logits2 = model(data_verflip)
          logits3 = model(data_horflip)
          logits = (logits1 + logits2 + logits3) / 3
          loss = nn.functional.cross_entropy(logits, target)

        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        loss_log.append(loss.item())
        acc_log.append(acc.item())

    wandb.log({"test_acc": np.mean(acc_log), "test_loss": np.mean(loss_log)})

def train_epoch_with_wandb(model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.flatten().to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        loss_log.append(loss.item())
        acc_log.append(acc.item())

    wandb.log({"train_acc": np.mean(acc_log), "train_loss": np.mean(loss_log)})

def train_with_wandb_and_test_augs(model, optimizer, n_epochs, train_loader, val_loader, model_name, scheduler=None):
    wandb.login(key='46c3b8e339b3fb22dc286204510c8af5b2c3e2e5')
    wandb.init(
        project='DL_BHW_1',
        entity='bspanfilov',
        name='first run'
    )

    for epoch in range(n_epochs):
        train_epoch_with_wandb(model, optimizer, train_loader)
        test_with_wandb_and_augs(model, val_loader)

        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"/kaggle/working/epoch_{epoch}_model_{model_name}.pt")
            torch.save(optimizer.state_dict(), f"/kaggle/working/epoch_{epoch}optimizer_{model_name}.pt")
        print(f"Epoch {epoch}")

        if scheduler is not None:
            scheduler.step()

    wandb.finish()

def train_with_wandb(model, optimizer, n_epochs, train_loader, val_loader, model_name, scheduler=None):
    wandb.login(key='46c3b8e339b3fb22dc286204510c8af5b2c3e2e5')
    wandb.init(
        project='DL_BHW_1',
        entity='bspanfilov',
        name='first run'
    )

    for epoch in range(n_epochs):
        train_epoch_with_wandb(model, optimizer, train_loader)
        test_with_wandb(model, val_loader)

        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"/kaggle/working/epoch_{epoch}_model_{model_name}.pt")
            torch.save(optimizer.state_dict(), f"/kaggle/working/epoch_{epoch}optimizer_{model_name}.pt")
        print(f"Epoch {epoch}")

        if scheduler is not None:
            scheduler.step()

    wandb.finish()
