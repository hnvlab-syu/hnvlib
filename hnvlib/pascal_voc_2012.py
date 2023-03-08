# implemented and written by Yeoreum Lee, Wangtaek Oh in AI HnV Lab @ Sahmyook University in 2023
__author__ = 'leeyeoreum02, ohkingtaek'

import os
import shutil
import random
from glob import glob
from typing import Optional, Sequence, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import deeplabv3_resnet50
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import MulticlassJaccardIndex


torch.set_float32_matmul_precision('medium')


NUM_CLASSES = 20


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


class PascalVOC2012Dataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
        mask_transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()
        
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.image_ids = df['image_id'].tolist()

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tensor:
        image_id = self.image_ids[index]

        image = Image.open(os.path.join(self.image_dir, f'{image_id}.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.label_dir, f'{image_id}.png'))

        mask = np.asarray(mask)
        height, width = mask.shape

        target = np.zeros((height, width, NUM_CLASSES+1))
        for class_id in range(NUM_CLASSES):
            target[mask == class_id+1, class_id+1] = 1

        if self.transform is not None:
            image = self.transform(image)

        if self.mask_transform is not None:
            target = self.mask_transform(target)

        meta_data = {
            'image_id': image_id,
            'height': height,
            'width': width
        }

        return image, target, meta_data
    

def get_transform(size: Sequence[int]):
    return Compose([
        ToTensor(),
        Resize(size),
    ])


def get_mask_transform(size: Sequence[int]):
    return Compose([
        ToTensor(),
        Resize(size, interpolation=InterpolationMode.NEAREST),
    ])


def visualize_dataset(
    image_dir: os.PathLike,
    label_dir: os.PathLike,
    csv_path: os.PathLike,
    size: Sequence[int],
    save_dir: os.PathLike,
    n_images: int = 10,
    alpha: float = 0.5
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

    dataset = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        image, target, meta_data = dataset[i]
        image = (image * 255.0).type(torch.uint8)

        result = draw_segmentation_masks(image, target.type(torch.bool), alpha=alpha)
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{meta_data['image_id']}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def train(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)['out']
        preds = torch.softmax(preds, dim=1)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss = loss.item()
            current = batch * len(images)
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
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)['out']
            preds = torch.softmax(preds, dim=1)

            test_loss += loss_fn(preds, targets).item()
            metric.update(preds, targets.argmax(dim=1))
    test_loss /= num_batches
    miou = metric.compute()
    print(f'Test Error: \n mIoU: {(100*miou):>0.1f}, Avg loss: {test_loss:>8f} \n')
    
    metric.reset()
    print()


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, n_images: int = 10, alpha: float = 0.5) -> None:
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
        image, _, meta_data = testset[i]
        image_id, height, width = meta_data.values()

        image = image.to(device)
        pred = model(image.unsqueeze(0))['out']
        pred = torch.softmax(pred, dim=1)

        max_index = torch.argmax(pred, dim=1)
        pred_bool = torch.zeros_like(pred, dtype=torch.bool).scatter(1, max_index.unsqueeze(1), True)

        image = (image * 255.0).type(torch.uint8)
        result = draw_segmentation_masks(image.cpu(), pred_bool.cpu().squeeze(), alpha=alpha)
        result = F.resize(result, size=(height, width))
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def run_pytorch(
    image_dir: os.PathLike,
    label_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float,
    size: Sequence[int]
) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    split_dataset(label_dir)

    visualize_dataset(image_dir, label_dir, train_csv_path, size, save_dir='examples/pascal-voc-2012/train', alpha=0.8)
    visualize_dataset(image_dir, label_dir, test_csv_path, size, save_dir='examples/pascal-voc-2012/test', alpha=0.8)

    training_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=train_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = deeplabv3_resnet50(num_classes=NUM_CLASSES+1).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES+1, ignore_index=0).to(device)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model, loss_fn, metric)
    print('Done!')

    torch.save(model.state_dict(), 'pascal-voc-2012-deeplabv3.pth')
    print('Saved PyTorch Model State to pascal-voc-2012-deeplabv3.pth')

    model = deeplabv3_resnet50(num_classes=NUM_CLASSES+1)
    model.load_state_dict(torch.load('pascal-voc-2012-deeplabv3.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/pascal-voc-2012/deeplabv3', alpha=0.8)


# ====================== PyTorch Lightning ======================


class PascalVOC2012Module(pl.LightningModule):
    """모델과 학습/추론 코드가 포함된 파이토치 라이트닝 모듈입니다.
    """
    def __init__(self, lr: Optional[float] = None) -> None:
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=NUM_CLASSES+1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES+1, ignore_index=0)

        self.lr = lr if self.lr is not None else 1e-2

    def configure_optimizers(self):
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

    def training_step(self, batch: Tensor, batch_idx: int):
        """뉴럴 네트워크를 한 스텝 훈련합니다.

        :param batch: 훈련 데이터셋의 배치 크기
        :type batch: int
        :param batch_idx: 배치에 대한 인덱스
        :type batch_idx: int
        :return: 훈련 오차 데이터
        :rtype: Dict[str, float]
        """
        images, targets, _ = batch

        preds = self.model(images)['out']
        preds = torch.softmax(preds, dim=1)
        loss = self.loss_fn(preds, targets)

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
        images, targets, _ = batch

        preds = self.model(images)['out']
        preds = torch.softmax(preds, dim=1)
        loss = self.loss_fn(preds, targets)
        self.metric.update(preds, targets.argmax(dim=1))

        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        """한 에폭 검증을 마치고 실행되는 코드입니다.
        
        :param outputs: 함수 validation_step에서 반환한 값들을 한 에폭이 끝나는 동안 모은 값들의 집합
        :type outputs: List[Tensor]
        """
        self.log('val_miou', self.metric.compute(), prog_bar=True)
        self.metric.reset()


def run_pytorch_lightning(
    root_dir: os.PathLike,
    image_dir: os.PathLike,
    label_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float,
    size: Sequence[int],
) -> None:
    split_dataset(label_dir, root_dir)

    visualize_dataset(image_dir, label_dir, train_csv_path, size, save_dir='examples/pascal-voc-2012/train', alpha=0.8)
    visualize_dataset(image_dir, label_dir, test_csv_path, size, save_dir='examples/pascal-voc-2012/test', alpha=0.8)

    training_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=train_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    test_data = PascalVOC2012Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        csv_path=test_csv_path,
        transform=get_transform(size=size),
        mask_transform=get_mask_transform(size=size)
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    model = PascalVOC2012Module(lr=lr)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('pascal-voc-2012-deeplabv3.ckpt')
    print('Saved PyTorch Lightning Model State to pascal-voc-2012-deeplabv3.ckpt')

    model = PascalVOC2012Module.load_from_checkpoint(checkpoint_path='pascal-voc-2012-deeplabv3.ckpt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    visualize_predictions(test_data, device, model, 'examples/pascal-voc-2012/deeplapv3-lightning', alpha=0.8)
