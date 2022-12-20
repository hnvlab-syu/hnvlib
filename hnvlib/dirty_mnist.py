import os
import csv
import random
from typing import Dict, Sequence, Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics import Accuracy


def split_dataset(path: os.PathLike, split_rate: float = 0.2) -> None:
    df = pd.read_csv(path)
    size = len(df)
    indices = list(range(size))
    random.shuffle(indices)

    split_point = int(split_rate * size)
    test_df = df.loc[indices[:split_point]]
    test_df.to_csv('data/dirty-mnist/test_answer.csv', index=False)
    train_df = df.loc[indices[split_point:]]
    train_df.to_csv('data/dirty-mnist/train_answer.csv', index=False)


class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id))

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x


def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader, device, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = pred > 0.5
            correct += (pred == y).float().mean().item()
    test_loss /= num_batches
    correct /= num_batches
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def predict(test_data, model):
    from string import ascii_lowercase
    classes = list(ascii_lowercase)

    model.eval()
    x = test_data[1][0].unsqueeze(0)
    y = test_data[1][1]
    with torch.no_grad():
        preds = model(x) > 0.5
        print(preds)
        preds = preds.squeeze().nonzero()
        actual = y.nonzero()[0]
        print(f'Predicted: "{[classes[pred] for pred in preds]}", Actual: "{[classes[a] for a in actual]}"')


def run_pytorch(batch_size, epochs):
    split_dataset('data/dirty-mnist/dirty_mnist_2nd_answer.csv')

    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    trainset = MnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/train_answer.csv',
        transforms_train
    )
    testset = MnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/test_answer.csv',
        transforms_test
    )

    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MnistModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MultiLabelSoftMarginLoss()

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model, loss_fn)
    print('Done!')

    torch.save(model.state_dict(), 'dirty-mnist.pth')
    print('Saved PyTorch Model State to dirty-mnist.pth')

    model = MnistModel()
    model.load_state_dict(torch.load('dirty-mnist.pth'))
    predict(testset, model)


class MnistModule(pl.LightningModule):
    def __init__(self) -> None:
        super(MnistModule, self).__init__()
        self.model = MnistModel()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.metric = Accuracy(num_classes=26)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        X, y = batch

        pred = self(X)
        loss = self.loss_fn(pred, y)

        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        X, y = batch

        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.metric.update(pred, y)

        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', self.metric.compute(), prog_bar=True)
        self.metric.reset()


def run_pytorch_lightning(batch_size, epochs):
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    trainset = MnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/train_answer.csv',
        transforms_train
    )
    testset = MnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/test_answer.csv',
        transforms_test
    )

    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    model = MnistModule()
    trainer = Trainer(max_epochs=epochs, gpus=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('dirty-mnist.ckpt')
    print('Saved PyTorch Lightning Model State to dirty-mnist.ckpt')

    model = MnistModule.load_from_checkpoint(checkpoint_path='dirty-mnist.ckpt')
    predict(testset, model)
