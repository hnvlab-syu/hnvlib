"""CIFAR-10 데이터셋으로 간단한 뉴럴 네트워크를 훈련하고 추론하는 코드입니다.
CIFAR-10 Dataset Link : https://www.cs.toronto.edu/~kriz/cifar.html
"""
# implemented and written by Yeoreum Lee, Wangtaek Oh in AI HnV Lab @ Sahmyook University in 2023
__author__ = 'leeyeoreum02, ohkingtaek'

from typing import Dict, List, TypeVar, Optional

import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy


NUM_CLASSES = 10


_device = TypeVar('_device')
_Optimizer = torch.optim.Optimizer


class LeNet(nn.Module):
    """학습과 추론에 사용되는 간단한 뉴럴 네트워크입니다.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워드(순전파)를 진행하는 함수입니다.

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값 (클래스값)
        :rtype: Tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataloader: DataLoader, device: _device, model: nn.Module, loss_fn: nn.Module, optimizer: _Optimizer) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크를 훈련합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: _device
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader: DataLoader, device: _device, model: nn.Module, loss_fn: nn.Module) -> None:
    """CIFAR-10 데이터셋으로 뉴럴 네트워크의 성능을 테스트합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 테스트에 사용되는 장치
    :type device: _device
    :param model: 테스트에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 테스트에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def predict(test_data: Dataset, model: nn.Module) -> None:
    """학습한 뉴럴 네트워크로 CIFAR-10 데이터셋을 분류합니다.

    :param test_data: 추론에 사용되는 데이터셋
    :type test_data: Dataset
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    """
    classes = [
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    ]

    model.eval()
    image = test_data[0][0].unsqueeze(0)
    target = test_data[0][1]
    with torch.no_grad():
        pred = model(image)
        predicted = classes[pred[0].argmax(0)]
        actual = classes[target]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def run_pytorch(batch_size: int, epochs: int, lr: float) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LeNet(num_classes=NUM_CLASSES).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model, loss_fn)
    print('Done!')

    torch.save(model.state_dict(), 'cifar-net-lenet.pth')
    print('Saved PyTorch Model State to cifar-net-lenet.pth')

    model = LeNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('cifar-net-lenet.pth'))

    predict(test_data, model)


# ====================== PyTorch Lightning ======================


class CIFARNetworkModule(pl.LightningModule):
    """모델과 학습/추론 코드가 포함된 파이토치 라이트닝 모듈입니다.
    """
    def __init__(self, lr: Optional[float] = None) -> None:
        super().__init__()
        self.model = LeNet(num_classes=NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = Accuracy(num_classes=NUM_CLASSES)
        
        self.lr = lr if lr is not None else 0.01

    def configure_optimizers(self) -> _Optimizer:
        """옵티마이저를 정의합니다.
        
        :return: 파이토치 옵티마이저
        :rtype: torch.optim.Optimizer
        """
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워딩 함수

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
        images, targets = batch

        preds = self(images)
        loss = self.loss_fn(preds, targets)

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
        images, targets = batch

        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.metric.update(preds, targets)

        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        """한 에폭 검증을 마치고 실행되는 코드입니다.
        
        :param outputs: 함수 validation_step에서 반환한 값들을 한 에폭이 끝나는 동안 모은 값들의 집합
        :type outputs: List[Tensor]
        """
        self.log('val_acc', self.metric.compute(), prog_bar=True)
        self.metric.reset()


def run_pytorch_lightning(batch_size: int, epochs: int, lr: float) -> None:
    """학습/추론 파이토치 라이트닝 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    model = CIFARNetworkModule(lr=lr)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('cifar-net-lenet.ckpt')
    print('Saved PyTorch Lightning Model State to cifar-net-lenet.ckpt')

    model = CIFARNetworkModule.load_from_checkpoint(checkpoint_path='cifar-net-lenet.ckpt')
    
    predict(test_data, model)
