"""Wheat 데이터셋으로 간단한 뉴럴 네트워크를 훈련하고 추론하는 코드입니다.
Wheat Dataset Link : https://www.kaggle.com/c/global-wheat-detection
"""

from collections import defaultdict
import os
import shutil
from typing import Callable, Optional, Sequence, Tuple
import random
import re

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import ToTensor, Compose
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_lightning import LightningModule


TRAIN_IMAGE_DIR = '../../../data/global-wheat-detection/train'
TRAIN_CSV_PATH = '../../../data/global-wheat-detection/train.csv'
TEST_IMAGE_DIR = '../../../data/global-wheat-detection/test'
TEST_CSV_PATH = '../../../data/global-wheat-detection/sample_submission.csv'


class WheatDataset(Dataset):
    """Wheat 데이터셋 사용자 정의 클래스를 정의합니다.
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
        is_test: bool = False
    ) -> None:
        """데이터 정보를 불러와 정답(bbox)과 각각 데이터의 이름(image_id)를 저장
        
        :param image_dir: 데이터셋 경로
        :type image_dir: os.PathLike
        :param csv_path: 데이터셋 정보를 담고있는 csv 파일 경로
        :type csv_path: os.PathLike
        :param transform: 데이터셋을 정규화하거나 텐서로 변환, augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transform: Optional[Sequence[Callable]]
        :param is_test: 테스트 데이터인지 아닌지 확인
        :type is_test: bool
        """
        super().__init__()
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_ids = df['image_id'].unique()

        self.is_test = is_test

        if not self.is_test:
            self.all_bboxes = defaultdict(list)
            for _, row in df.iterrows():
                bbox = re.sub('\[|\]', '', row['bbox'])
                bbox = list(map(float, bbox.split(', ')))
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]
                self.all_bboxes[row['image_id']].append(bbox)

        self.transform = transform

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
        :return: 이미지 한장과 정답 {bbox, labels}를 같이 반환
        :rtype: Tuple[Tensor]
        """
        image_id = self.image_ids[index]

        image = Image.open(os.path.join(self.image_dir, f'{image_id}.jpg')).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image

        bboxes_per_image = self.all_bboxes[image_id]
        class_labels = [1] * len(bboxes_per_image)

        target = {
            'boxes': torch.as_tensor(bboxes_per_image, dtype=torch.float32),
            'labels': torch.as_tensor(class_labels, dtype=torch.int64)
        }

        return image, target


def visualize_dataset(save_dir: os.PathLike) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = WheatDataset(
        image_dir=TRAIN_IMAGE_DIR,
        csv_path=TRAIN_CSV_PATH,
    )

    indices = random.Random(36).choices(range(len(dataset)), k=5)
    for i in indices:
        image, target = dataset[i]
        image = np.array(image)
        image_id = dataset.image_ids[i]

        plt.imshow(image)
        ax = plt.gca()

        for x1, y1, x2, y2 in target['boxes']:
            w = x2 - x1
            h = y2 - y1

            category_id = 'wheat'

            rect = patches.Rectangle(
                (x1, y1),
                w, h,
                linewidth=1,
                edgecolor='green',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1,
                category_id,
                c='white',
                size=6,
                path_effects=[pe.withStroke(linewidth=2, foreground='green')],
                family='sans-serif',
                weight='semibold',
                va='top', ha='left',
                bbox=dict(
                    boxstyle='round',
                    ec='green',
                    fc='green',
                )
            )

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def get_transform() -> Sequence[Callable]:
    """데이터셋을 Tensor로 변환해주는 함수 Sequence, 임의로 추가 가능
    
    :return: 함수들의 Sequence
    :rtype: Sequence[Callable]
    """
    return Compose([
        ToTensor()
    ])


def collate_fn(batch: Tensor) -> Tuple:
    return tuple(zip(*batch))


def train(dataloader: Dataloader, device: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Wheat 데이터셋으로 뉴럴 네트워크를 훈련합니다.
    
    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(images)
            print(f'total loss: {losses:>4f}, cls loss: {loss_dict["loss_classifier"]:>4f}, box loss: {loss_dict["loss_box_reg"]:>4f} [{current:>5d}/{size:>5d}]')


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, conf_thr: float = 0.1) -> None:
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
    for i, image in enumerate(testset):
        image = [image.to(device)]
        preds = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        # image = image * 255

        image_id = testset.image_ids[i]

        plt.imshow(image)
        ax = plt.gca()

        preds = [{k: v.detach().cpu() for k, v in t.items()} for t in preds]
        for score, (x1, y1, x2, y2) in zip(preds[0]['scores'], preds[0]['boxes']):
            if score >= conf_thr:
                w = x2 - x1
                h = y2 - y1

                category_id = 'wheat'

                rect = patches.Rectangle(
                    (x1, y1),
                    w, h,
                    linewidth=1,
                    edgecolor='green',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1,
                    f'{category_id}: {score:.2f}',
                    c='white',
                    size=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground='green')],
                    family='sans-serif',
                    weight='semibold',
                    va='top', ha='left',
                    bbox=dict(
                        boxstyle='round',
                        ec='green',
                        fc='green',
                    )
                )

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def run_pytorch(batch_size: int, epochs: int) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    visualize_dataset('../examples/train_dataset')

    transform = get_transform()

    trainset = WheatDataset(
        image_dir=TRAIN_IMAGE_DIR,
        csv_path=TRAIN_CSV_PATH,
        transform=transform
    )
    testset = WheatDataset(
        image_dir=TEST_IMAGE_DIR,
        csv_path=TEST_CSV_PATH,
        transform=transform,
        is_test=True
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = fasterrcnn_resnet50_fpn(num_classes=1+1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # for t in range(epochs):
    #     print(f'Epoch {t+1}\n-------------------------------')
    #     train(trainloader, device, model, optimizer)
    # print('Done!')

    # torch.save(model.state_dict(), 'wheat-faster-rcnn.pth')
    # print('Saved PyTorch Model State to wheat-faster-rcnn.pth')

    model = fasterrcnn_resnet50_fpn(num_classes=1+1).to(device)
    model.load_state_dict(torch.load('wheat-faster-rcnn.pth'))
    visualize_predictions(testset, device, model, '../examples/faster-rcnn')
    
    
def run_pytorch_lightning():
    class WheatDetectionModule(LightningModule):
        def __init__(self, num_classes):
            super().__init__()
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            images, targets = batch
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            return {'loss': losses, 'log': {'train_loss': losses}}
        
        def validation_step(self, batch, batch_idx):
            images, targets = batch
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            return {'val_loss': losses}
        
        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            return {'avg_val_loss': avg_loss}
        
        def configure_optimizers(self):
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            return [optimizer], [scheduler]
        
        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
        
        def val_dataloader(self):
            return DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)


def main():
    run_pytorch(16, 5)


if __name__ == '__main__':
    main()
