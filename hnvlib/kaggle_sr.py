import os
import math
import random
import shutil
from glob import glob
from typing import Optional, Sequence, Callable

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import PeakSignalNoiseRatio


torch.set_float32_matmul_precision('medium')


def split_dataset(label_dir: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    root_dir = os.path.dirname(label_dir)

    image_ids = []
    for path in glob(os.path.join(label_dir, '*.png')):
        file_name = os.path.split(path)[-1]
        image_id = os.path.splitext(file_name)[0]
        image_ids.append(image_id)

    random.shuffle(image_ids)

    split_point = int(split_rate * len(image_ids))

    test_ids = image_ids[:split_point]
    train_ids = image_ids[split_point:]

    test_df = pd.DataFrame({'image_id': test_ids})
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.DataFrame({'image_id': train_ids})
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


class KaggleSRDataset(Dataset):
    def __init__(
        self,
        lr_dir: os.PathLike,
        hr_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()
        
        df = pd.read_csv(csv_path)
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.image_ids = df['image_id'].tolist()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tensor:
        image_id = self.image_ids[index]

        lr_image = Image.open(os.path.join(self.lr_dir, f'{image_id}.png')).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, f'{image_id}.png')).convert('RGB')

        if self.transform is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image, image_id
    

def visualize_dataset(
    lr_dir: os.PathLike,
    hr_dir: os.PathLike,
    csv_path: os.PathLike,
    save_dir: os.PathLike,
    n_images: int = 10,
) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=csv_path,
        transform=transforms.ToTensor()
    )

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        lr_image, hr_image, image_id = dataset[i]
        _, lr_h, lr_w = lr_image.shape
        _, hr_h, hr_w = hr_image.shape

        background = torch.ones_like(hr_image)
        edge_width = (hr_w - lr_w) // 2
        edge_height = (hr_h - lr_h) // 2
        background[:, edge_height:edge_height+lr_h, edge_width:edge_width+lr_w] = lr_image
        background = (background * 255.0).type(torch.uint8)
        hr_image = (hr_image * 255.0).type(torch.uint8)
        
        _, axs = plt.subplots(ncols=2, squeeze=False)
        for i, img in enumerate([background, hr_image]):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(32)]
        )
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.upscale(x)
        return x


def train(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (lr_images, hr_images, _) in enumerate(dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        preds = model(lr_images)
        loss = loss_fn(preds, hr_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss = loss.item()
            current = batch * len(lr_images)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, metric) -> None:
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
    with torch.no_grad():
        for lr_images, hr_images, _ in dataloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            preds = model(lr_images)

            test_loss += loss_fn(preds, hr_images).item()
            metric.update(preds, hr_images)
    test_loss /= num_batches
    psnr = metric.compute()
    print(f'Test Error: \n PSNR: {psnr:>0.1f}, Avg loss: {test_loss:>8f} \n')

    metric.reset()
    print()


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, n_images: int = 10) -> None:
    """이미지에 bbox 그려서 저장 및 시각화
    
    :param testset: 추론에 사용되는 데이터셋
    :type testset: Dataset
    :param device: 추론에 사용되는 장치
    :type device: str
    :param model: 추론에 사용되는 모델
    :type model: nn.Module
    :param save_dir: 추론한 사진이 저장되는 경로
    :type save_dir: os.PathLike
    :param conf_thr: confidence threshold - 해당 숫자에 만족하지 않는 bounding box 걸러내는 파라미터
    :type conf_thr: float
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        lr_image, hr_image, image_id = testset[i]

        lr_image = lr_image.to(device)
        pred = model(lr_image.unsqueeze(0))

        pred = (pred.squeeze(0) * 255.0).type(torch.uint8)
        hr_image = (hr_image * 255.0).type(torch.uint8)
        
        fig, axs = plt.subplots(ncols=2, squeeze=False)
        for i, img in enumerate([hr_image, pred]):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


def run_pytorch(
    root_dir: os.PathLike,
    lr_dir: os.PathLike,
    hr_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float
) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    split_dataset(hr_dir)

    visualize_dataset(lr_dir, hr_dir, train_csv_path, save_dir='examples/kaggle-sr/train')
    visualize_dataset(lr_dir, hr_dir, test_csv_path, save_dir='examples/kaggle-sr/test')

    training_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=train_csv_path,
        transform=transforms.ToTensor()
    )

    test_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EDSR().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = PeakSignalNoiseRatio().to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model, loss_fn, metric)
    print('Done!')

    torch.save(model.state_dict(), 'kaggle-sr-edsr.pth')
    print('Saved PyTorch Model State to kaggle-sr-edsr.pth')

    model = EDSR()
    model.load_state_dict(torch.load('kaggle-sr-edsr.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/kaggle-sr/edsr')


class KaggleSRModule(pl.LightningModule):
    """모델과 학습/추론 코드가 포함된 파이토치 라이트닝 모듈입니다.
    """
    def __init__(self) -> None:
        super(KaggleSRModule, self).__init__()
        self.model = EDSR()
        self.loss_fn = nn.MSELoss()
        self.metric = PeakSignalNoiseRatio()

    def configure_optimizers(self):
        """옵티마이저를 정의합니다.
        
        :return: 파이토치 옵티마이저
        :rtype: torch.optim.Optimizer
        """
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x: Tensor) -> Tensor:
        """피드 포워딩 함수

        :param x: 입력 이미지
        :type x: Tensor
        :return: 입력 이미지에 대한 예측값
        :rtype: Tensor
        """
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int):
        """뉴럴 네트워크를 한 스텝 훈련합니다.

        :param batch: 훈련 데이터셋의 배치 크기
        :type batch: int
        :param batch_idx: 배치에 대한 인덱스
        :type batch_idx: int
        :return: 훈련 오차 데이터
        :rtype: Dict[str, float]
        """
        lr_images, hr_images, _ = batch

        pred = self.model(lr_images)
        loss = self.loss_fn(pred, hr_images)

        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch: Tensor, batch_idx: int):
        """훈련 후 한 배치를 검증합니다.

        :param batch: 검증 데이터셋의 배치 크기
        :type batch: int
        :param batch_idx: 배치에 대한 인덱스
        :type batch_idx: int
        :return: 검증 오차 데이터
        :rtype: Dict[str, float]
        """
        lr_images, hr_images, _ = batch

        pred = self.model(lr_images)
        loss = self.loss_fn(pred, hr_images)
        self.metric.update(pred, hr_images)

        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        """한 에폭 검증을 마치고 실행되는 코드입니다.
        
        :param outputs: 함수 validation_step에서 반환한 값들을 한 에폭이 끝나는 동안 모은 값들의 집합
        :type outputs: List[Tensor]
        """
        self.log('val_psnr', self.metric.compute(), prog_bar=True)
        self.metric.reset()


def run_pytorch_lightning(
    root_dir: os.PathLike,
    lr_dir: os.PathLike,
    hr_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
) -> None:
    split_dataset(hr_dir, root_dir)

    visualize_dataset(lr_dir, hr_dir, train_csv_path, save_dir='examples/kaggle-sr/train')
    visualize_dataset(lr_dir, hr_dir, test_csv_path, save_dir='examples/kaggle-sr/test')

    training_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=train_csv_path,
        transform=transforms.ToTensor()
    )

    test_data = KaggleSRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        csv_path=test_csv_path,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    model = KaggleSRModule()
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('kaggle_sr_edsr.ckpt')
    print('Saved PyTorch Lightning Model State to kaggle_sr_edsr.ckpt')

    model = KaggleSRModule.load_from_checkpoint(checkpoint_path='kaggle_sr_edsr.ckpt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    visualize_predictions(test_data, device, model, 'examples/kaggle-sr/edsr-lightning')
