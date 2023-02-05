"""Wheat 데이터셋으로 간단한 뉴럴 네트워크를 훈련하고 추론하는 코드입니다.
Wheat Dataset Link : https://www.kaggle.com/c/global-wheat-detection
"""

from collections import defaultdict
import os
import shutil
from typing import Callable, Optional, Sequence, Tuple, TypeVar
import random
import re
from pprint import pprint
import json
from ast import literal_eval

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torchvision
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import ToTensor, Compose
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


_device = TypeVar('_device')


torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')


def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)

    grouped = df.groupby(by='image_id')
    grouped_list = [group for _, group in grouped]

    split_point = int(split_rate * len(grouped))

    save_dir = os.path.split(csv_path)[0]
    test_df = pd.concat(grouped_list[:split_point])
    test_df.to_csv(os.path.join(save_dir, 'test_answer.csv'), index=False)
    train_df = pd.concat(grouped_list[split_point:])
    train_df.to_csv(os.path.join(save_dir, 'train_answer.csv'), index=False)


class WheatDataset(Dataset):
    """Wheat 데이터셋 사용자 정의 클래스를 정의합니다.
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
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

        bboxes_per_image = self.all_bboxes[image_id]
        class_labels = [1] * len(bboxes_per_image)

        target = {
            'boxes': torch.as_tensor(bboxes_per_image, dtype=torch.float32),
            'labels': torch.as_tensor(class_labels, dtype=torch.int64)
        }

        return image, target, image_id
    

def collate_fn(batch: Tensor) -> Tuple:
    return tuple(zip(*batch))


def visualize_dataset(image_dir: os.PathLike, csv_path: os.PathLike, save_dir: os.PathLike, n_images: int = 10) -> None:
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
        image_dir=image_dir,
        csv_path=csv_path,
    )

    indices = random.Random(36).choices(range(len(dataset)), k=n_images)
    for i in indices:
        image, target, image_id = dataset[i]
        image = np.array(image)

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


def train(dataloader: DataLoader, device: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
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
    for batch, (images, targets, _) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(images)
            print(f'total loss: {loss:>4f}, cls loss: {loss_dict["loss_classifier"]:>4f}, box loss: {loss_dict["loss_box_reg"]:>4f}, obj loss: {loss_dict["loss_objectness"]:>4f}, rpn loss: {loss_dict["loss_rpn_box_reg"]:>4f} [{current:>5d}/{size:>5d}]')


class MeanAveragePrecision:
    def __init__(self, csv_path: os.PathLike) -> None:
        self.id_csv2coco = {}
        json_path = self.to_coco(csv_path)
        self.coco_gt = COCO(json_path)

        self.preds = []

    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)
        
        grouped = df.groupby(by='image_id')
        grouped_dict = {image_id: group for image_id, group in grouped}
        
        res = defaultdict(list)

        n_id = 0
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            res['images'].append({
                'id': image_id,
                'width': 1024,
                'height': 1024,
                'file_name': f'{file_name}.jpg',
            })

            self.id_csv2coco[file_name] = image_id

            for _, row in group.iterrows():
                x1, y1, w, h = literal_eval(row['bbox'])
                res['annotations'].append({
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,
                })
                n_id += 1

        res['categories'].extend([{'id': 0, 'name': '_background_'}, {'id': 1, 'name': 'wheat'}])
            
        root_dir = os.path.split(csv_path)[0]
        save_path = os.path.join(root_dir, 'coco_annotations.json')
        with open(save_path, 'w') as f:
            json.dump(res, f)

        return save_path
    
    def update(self, preds, image_ids):
        self.preds.extend(list(zip(preds, image_ids)))

    def reset(self):
        self.preds = []

    def compute(self):
        detections = []
        for p, image_id in self.preds:
            pred_boxes = p['boxes']
            pred_boxes[:, 2] = pred_boxes[:, 2] - pred_boxes[:, 0]
            pred_boxes[:, 3] = pred_boxes[:, 3] - pred_boxes[:, 1]
            pred_boxes = pred_boxes.cpu().numpy()

            pred_scores = p['scores'].cpu().numpy()
            pred_labels = p['labels'].cpu().numpy()

            image_id = self.id_csv2coco[image_id]
            for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
                dt = {
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'score': s
                }
                detections.append(dt)

        coco_dt = self.coco_gt.loadRes(detections)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_map = coco_eval.stats[0]

        return coco_map


def test(dataloader: DataLoader, device: _device, model: nn.Module, metric) -> None:
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
    num_batches = len(dataloader)
    test_loss = 0
    test_cls_loss = 0
    test_box_loss = 0
    test_obj_loss = 0
    test_rpn_loss = 0

    with torch.no_grad():
        for batch, (images, targets, image_ids) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            test_loss += loss
            test_cls_loss += loss_dict['loss_classifier']
            test_box_loss += loss_dict['loss_box_reg']
            test_obj_loss += loss_dict['loss_objectness']
            test_rpn_loss += loss_dict['loss_rpn_box_reg']

            model.eval()
            preds = model(images)
            # print(preds)
            metric.update(preds, image_ids)

    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches

    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n')
    metric.compute()
    metric.reset()
    print()


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, conf_thr: float = 0.1, n_images: int = 10) -> None:
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
    indices = random.Random(36).choices(range(len(testset)), k=n_images)
    for i in indices:
        image, target, image_id = testset[i]
        image = [image.to(device)]
        preds = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)

        plt.imshow(image)
        ax = plt.gca()

        preds = [{k: v.detach().cpu() for k, v in t.items()} for t in preds]
        for score, (x1, y1, x2, y2) in zip(preds[0]['scores'], preds[0]['boxes']):
            if score >= conf_thr:
                w = x2 - x1
                h = y2 - y1

                category_id = preds[0]['labels']

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


def run_pytorch(
    csv_path: os.PathLike,
    train_image_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float,
) -> None:
    """학습/추론 파이토치 파이프라인입니다.

    :param batch_size: 학습 및 추론 데이터셋의 배치 크기
    :type batch_size: int
    :param epochs: 전체 학습 데이터셋을 훈련하는 횟수
    :type epochs: int
    """
    split_dataset(csv_path=csv_path)
    
    visualize_dataset(image_dir=train_image_dir, csv_path=train_csv_path, save_dir='examples/global-wheat-detection/train')
    visualize_dataset(image_dir=train_image_dir, csv_path=test_csv_path, save_dir='examples/global-wheat-detection/test')

    trainset = WheatDataset(
        image_dir=train_image_dir,
        csv_path=train_csv_path,
        transform=ToTensor()
    )
    testset = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=ToTensor(),
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=1, collate_fn=collate_fn)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = fasterrcnn_resnet50_fpn(num_classes=1+1)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(csv_path=test_csv_path)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(trainloader, device, model, optimizer)
        print()
        test(testloader, device, model, metric)
    print('Done!')

    torch.save(model.state_dict(), 'wheat-faster-rcnn.pth')
    print('Saved PyTorch Model State to wheat-faster-rcnn.pth')

    model = fasterrcnn_resnet50_fpn(num_classes=1+1)
    model.load_state_dict(torch.load('wheat-faster-rcnn.pth'))
    model.to(device)

    visualize_predictions(testset, device, model, 'examples/global-wheat-detection/faster-rcnn')
    
    
class WheatDetectionModule(LightningModule):
    def __init__(self, csv_path, lr):
        super().__init__()

        self.lr = lr
        self.model = fasterrcnn_resnet50_fpn(num_classes=1+1)
        self.metric = MeanAveragePrecision(csv_path)
        
    def forward(self, x):
        self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict)

        return {'loss': loss, 'log': loss_dict}
    
    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.model.eval()
        preds = self.model(images)
        self.metric.update(preds, image_ids)

        self.log('val_loss', loss)
        self.log_dict(loss_dict)

        return {'val_loss': loss, 'log': loss_dict}
    
    def validation_epoch_end(self, outputs):
        self.metric.compute()
        self.metric.reset()
    
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005,
        )


def run_pytorch_lightning(
    csv_path: os.PathLike,
    train_image_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float,
) -> None:
    split_dataset(csv_path=csv_path)
    
    visualize_dataset(image_dir=train_image_dir, csv_path=train_csv_path, save_dir='examples/global-wheat-detection/train')
    visualize_dataset(image_dir=train_image_dir, csv_path=test_csv_path, save_dir='examples/global-wheat-detection/test')

    trainset = WheatDataset(
        image_dir=train_image_dir,
        csv_path=train_csv_path,
        transform=ToTensor()
    )
    testset = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=ToTensor(),
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=16, collate_fn=collate_fn)

    model = WheatDetectionModule(csv_path=test_csv_path, lr=lr)
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    trainer.save_checkpoint('wheat-faster-rcnn.ckpt')
    print('Saved PyTorch Lightning Model State to wheat-faster-rcnn.ckpt')

    model = WheatDetectionModule.load_from_checkpoint(checkpoint_path='wheat-faster-rcnn.ckpt')
    
    visualize_predictions(testset, model)
