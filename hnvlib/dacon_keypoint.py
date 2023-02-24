import os
import json
import shutil
import random
from collections import defaultdict
from typing import Optional, Sequence, Callable, Tuple, Dict

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.utils import draw_keypoints
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2


torch.set_float32_matmul_precision('medium')


NUM_CLASSES = 1
EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [6, 8], [8, 10],
    [5, 7], [7, 9], [5, 11], [11, 13], [13, 15], [6, 12],
    [12, 14], [14, 16], [5, 6], [0, 17], [5, 17], [6, 17],
    [11, 12]
]


def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    root_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)

    grouped = df.groupby(by='image')
    grouped_list = [group for _, group in grouped]

    split_point = int(split_rate * len(grouped))

    test_ids = grouped_list[:split_point]
    train_ids = grouped_list[split_point:]

    test_df = pd.concat(test_ids)
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)
    train_df = pd.concat(train_ids)
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)


class DaconKeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')
        image = np.asarray(image)

        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transform is not None:
            targets = self.transform(**targets)

            image = targets['image']
            image = image / 255.0

            targets = {
                'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
                'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
                'keypoints': torch.as_tensor(
                    np.concatenate([targets['keypoints'], np.ones((24, 1))], axis=1)[np.newaxis], dtype=torch.float32
                )
            }

        return image, targets, image_id
    

def get_transform():
    return A.Compose(
        [
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )
    

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

    dataset = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=get_transform()
    )

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        image, target, image_id = dataset[i]
        image = (image * 255.0).type(torch.uint8)

        result = draw_keypoints(image, target['keypoints'], connectivity=EDGES, colors='blue', radius=4, width=3)
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, image_id), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


def train(dataloader: DataLoader, device: str, model: nn.Module, optimizer: torch.optim.Optimizer):
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
            message = 'total loss: {:>4f}, cls loss: {:>4f}, box loss: {:>4f}, obj loss: {:>4f}, rpn loss: {:>4f}, kpt loss: {:>4f}  [{:>5d}/{:>5d}]'
            message = message.format(
                loss,
                loss_dict['loss_classifier'],
                loss_dict['loss_box_reg'],
                loss_dict['loss_objectness'],
                loss_dict['loss_rpn_box_reg'],
                loss_dict['loss_keypoint'],
                current,
                size
            )
            print(message)


class ObjectKeypointSimilarity:
    def __init__(self, image_dir: os.PathLike, csv_path: os.PathLike) -> None:
        self.image_dir = image_dir
        self.id_csv2coco = {}
        json_path = self.to_coco(csv_path)
        self.coco_gt = COCO(json_path)

        self.detections = []

    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)
    
        grouped = df.groupby(by='image')
        grouped_dict = {image_id: group for image_id, group in grouped}
        
        res = defaultdict(list)

        n_id = 0
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            with Image.open(os.path.join(self.image_dir, file_name), 'r') as image:
                width, height = image.size
            res['images'].append({
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name,
            })

            self.id_csv2coco[file_name] = image_id

            for _, row in group.iterrows():
                keypoints = row[1:].values.reshape(-1, 2)

                x1 = np.min(keypoints[:, 0])
                y1 = np.min(keypoints[:, 1])
                x2 = np.max(keypoints[:, 0])
                y2 = np.max(keypoints[:, 1])

                w = x2 - x1
                h = y2 - y1

                keypoints = np.concatenate([keypoints, np.ones((24, 1), dtype=np.int64)+1], axis=1).reshape(-1).tolist()
                res['annotations'].append({
                    'keypoints': keypoints,
                    'num_keypoints': 24,
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,
                })
                n_id += 1

        res['categories'].extend([
            {
                'id': 1,
                'name': 'person',
                'keypoints': df.keys()[1:].tolist(),
                'skeleton': EDGES,
            }
        ])
            
        root_dir = os.path.split(csv_path)[0]
        save_path = os.path.join(root_dir, 'coco_annotations.json')
        with open(save_path, 'w') as f:
            json.dump(res, f)

        return save_path
    
    def update(self, preds, image_ids):
        for p, image_id in zip(preds, image_ids):
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]
            p['boxes'] = p['boxes'].cpu().numpy()

            num_keypoints = len(p['keypoints'])
            p['keypoints'][:, :, 2] += 1
            p['keypoints'] = p['keypoints'].reshape(num_keypoints, -1) if num_keypoints > 0 else p['keypoints'] 
            p['keypoints'] = p['keypoints'].cpu().numpy()

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            image_id = self.id_csv2coco[image_id]
            for b, l, s, k in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'keypoints': k.tolist(),
                    'score': s
                })

    def reset(self):
        self.detections = []

    def compute(self):
        coco_dt = self.coco_gt.loadRes(self.detections)
        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.kpt_oks_sigmas = np.ones((24, 1)) * 0.05
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def test(dataloader: DataLoader, device, model: nn.Module, metric) -> None:
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
    test_kpt_loss = 0
    with torch.no_grad():
        for images, targets, image_ids in dataloader:
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
            test_kpt_loss += loss_dict['loss_keypoint']

            model.eval()
            preds = model(images)

            metric.update(preds, image_ids)
    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches
    test_kpt_loss /= num_batches
    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n Keypoint loss: {test_kpt_loss:>8f} \n')
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
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, image_id = testset[i]
        
        image = [image.to(device)]
        pred = model(image)
        pred = {k: v.detach().cpu() for k, v in pred[0].items() if pred[0]['scores'] >= conf_thr}

        image = (image * 255.0).type(torch.uint8)
        result = draw_keypoints(image.cpu(), pred['keypoints'], connectivity=EDGES, colors='blue', radius=4, width=3)
        plt.imshow(result.permute(1, 2, 0).numpy())

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, image_id), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def run_pytorch(
    csv_path: os.PathLike,
    image_dir: os.PathLike,
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
    split_dataset(csv_path)
    
    visualize_dataset(image_dir, train_csv_path, save_dir='examples/dacon-keypoint/train')
    visualize_dataset(image_dir, test_csv_path, save_dir='examples/dacon-keypoint/test')

    training_data = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=train_csv_path,
        transform=get_transform(),
    )
    test_data = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=test_csv_path,
        transform=get_transform(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, collate_fn=collate_fn)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = keypointrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1, num_keypoints=24).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = ObjectKeypointSimilarity(image_dir=image_dir, csv_path=test_csv_path)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, optimizer)
        test(test_dataloader, device, model, metric)
    print('Done!')

    torch.save(model.state_dict(), 'dacon-keypoint-rcnn.pth')
    print('Saved PyTorch Model State to dacon-keypoint-rcnn.pth')

    model = keypointrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1, num_keypoints=24)
    model.load_state_dict(torch.load('dacon-keypoint-rcnn.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/dacon-keypoint/keypoint-rcnn')


class DaconKeypointModule(pl.LightningModule):
    def __init__(self, image_dir, test_csv_path, lr: Optional[float] = None):
        """_summary_

        Args:
            csv_path (_type_): _description_
            lr (_type_): _description_
        """
        super().__init__()
        self.model = keypointrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1, num_keypoints=24)
        self.metric = ObjectKeypointSimilarity(image_dir, test_csv_path)
        
        self.lr = lr if lr is not None else 1e-2

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005,
        )
        
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict, prog_bar=True)

        return {'loss': loss, 'log': loss_dict}
    
    def validation_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.model.eval()
        preds = self.model(images)
        self.metric.update(preds, image_ids)

        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(loss_dict, prog_bar=True)

        return {'val_loss': loss, 'log': loss_dict}
    
    def validation_epoch_end(self, outputs):
        """_summary_

        Args:
            outputs (_type_): _description_
        """
        self.metric.compute()
        self.metric.reset()
    

def run_pytorch_lightning(
    csv_path: os.PathLike,
    image_dir: os.PathLike,
    train_csv_path: os.PathLike,
    test_csv_path: os.PathLike,
    batch_size: int,
    epochs: int,
    lr: float,
) -> None:
    """_summary_

    Args:
        csv_path (os.PathLike): _description_
        train_image_dir (os.PathLike): _description_
        train_csv_path (os.PathLike): _description_
        test_csv_path (os.PathLike): _description_
        batch_size (int): _description_
        epochs (int): _description_
        lr (float): _description_
    """
    split_dataset(csv_path)
    
    visualize_dataset(image_dir, train_csv_path, save_dir='examples/dacon-keypoint/train')
    visualize_dataset(image_dir, test_csv_path, save_dir='examples/dacon-keypoint/test')

    training_data = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=train_csv_path,
        transform=get_transform()
    )
    test_data = DaconKeypointDataset(
        image_dir=image_dir,
        csv_path=test_csv_path,
        transform=get_transform(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, collate_fn=collate_fn)

    model = DaconKeypointModule(image_dir=image_dir, test_csv_path=test_csv_path, lr=lr)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('dacon-keypoint-rcnn.ckpt')
    print('Saved PyTorch Lightning Model State to dacon-keypoint-rcnn.ckpt')

    model = DaconKeypointModule.load_from_checkpoint(checkpoint_path='dacon-keypoint-rcnn.ckpt', image_dir=image_dir, csv_path=test_csv_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    visualize_predictions(test_data, device, model, 'examples/dacon-keypoint/keypoint-rcnn-lightning')