"""월간 데이콘 제 2회 컴퓨터 비전 학습 경진대회 데이터셋으로 간단한 뉴럴 네트워크를 훈련하고 추론하는 코드입니다.
월간 데이콘 제 2회 컴퓨터 비전 학습 경진대회 Dataset Link : https://dacon.io/competitions/official/235697/overview/description
"""

import os
import csv
import random
from typing import Dict, Sequence, Callable, Tuple, List

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
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    df = pd.read_csv(path)
    size = len(df)
    indices = list(range(size))
    random.shuffle(indices)

    split_point = int(split_rate * size)
    test_df = df.loc[indices[:split_point]]
    test_df.to_csv('data/dirty-mnist/test_answer.csv', index=False)
    train_df = df.loc[indices[split_point:]]
    train_df.to_csv('data/dirty-mnist/train_answer.csv', index=False)


class DirtyMnistDataset(Dataset):
    """Dirty-MNIST 데이터셋 사용자 정의 클래스를 정의합니다.
    """
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장
        
        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환, augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

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
        """데이터셋의 길이를 반환
        
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수
        
        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값
        :rtype: Tuple[Tensor]
        """
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id))

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


class DirtyMnistModel(nn.Module):
    """Dirty-MNIST 데이터를 훈련할 모델을 정의합니다.
    모델은 torchvision에서 제공하는 ResNet-50을 지원합니다.
    Model Link : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워드(순전파)를 진행하는 함수입니다.

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값
        :rtype: Tensor
        """
        x = self.resnet(x)
        x = self.classifier(x)

        return x


def train(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Dirty-MNIST 데이터셋으로 뉴럴 네트워크를 훈련합니다.
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
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


def test(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
    """Dirty-MNIST 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
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


def predict(test_data: Dataset, model: nn.Module) -> None:
    """학습한 뉴럴 네트워크로 Dirty-MNIST 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
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


def run_pytorch(batch_size: int, epochs: int) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
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

    trainset = DirtyMnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/train_answer.csv',
        transforms_train
    )
    testset = DirtyMnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/test_answer.csv',
        transforms_test
    )

    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DirtyMnistModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MultiLabelSoftMarginLoss()

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model, loss_fn)
    print('Done!')

    torch.save(model.state_dict(), 'dirty-mnist.pth')
    print('Saved PyTorch Model State to dirty-mnist.pth')

    model = DirtyMnistModel()
    model.load_state_dict(torch.load('dirty-mnist.pth'))
    predict(testset, model)


class DirtyMnistModule(pl.LightningModule):
    """모델과 학습/추론 코드가 포함된 파이토치 라이트닝 모듈입니다.
    """
    def __init__(self) -> None:
        super(DirtyMnistModule, self).__init__()
        self.model = DirtyMnistModel()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.metric = Accuracy(num_classes=26)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """옵티마이저를 정의합니다.
        Adam Paper Link : https://arxiv.org/abs/1412.6980

        :return: 파이토치 옵티마이저
        :rtype: torch.optim.Optimizer
        """
        return optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워딩

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값
        :rtype: Tensor
        """
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        """뉴럴 네트워크를 한 스텝 훈련합니다.

        :param batch: 훈련 데이터셋의 배치 크기
        :type batch: int
        :param batch_idx: 배치에 대한 인덱스
        :type batch_idx: int
        :return: 훈련 오차 데이터
        :rtype: Dict[str, float]
        """
        X, y = batch

        pred = self(X)
        loss = self.loss_fn(pred, y)

        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        """훈련 후 한 배치를 검증합니다.

        :param batch: 검증 데이터셋의 배치 크기
        :type batch: int
        :param batch_idx: 배치에 대한 인덱스
        :type batch_idx: int
        :return: 검증 오차 데이터
        :rtype: Dict[str, float]
        """
        X, y = batch

        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.metric.update(pred, y)

        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        """한 에폭 검증을 마치고 실행되는 코드입니다.

        :param outputs: 함수 validation_step에서 반환한 값들을 한 에폭이 끝나는 동안 모은 값들의 집합
        :type outputs: List[Tensor]
        """
        self.log('val_acc', self.metric.compute(), prog_bar=True)
        self.metric.reset()


def run_pytorch_lightning(batch_size: int, epochs: int) -> None:
    """학습/추론 파이토치 라이트닝 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
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

    trainset = DirtyMnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/train_answer.csv',
        transforms_train
    )
    testset = DirtyMnistDataset(
        'data/dirty-mnist/train',
        'data/dirty-mnist/test_answer.csv',
        transforms_test
    )

    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    model = DirtyMnistModule()
    trainer = Trainer(max_epochs=epochs, gpus=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('dirty-mnist.ckpt')
    print('Saved PyTorch Lightning Model State to dirty-mnist.ckpt')

    model = DirtyMnistModule.load_from_checkpoint(checkpoint_path='dirty-mnist.ckpt')
    predict(testset, model)
