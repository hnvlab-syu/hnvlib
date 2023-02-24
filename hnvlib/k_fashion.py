import os
import json
import random
import shutil
from collections import defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as coco_mask

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


torch.set_float32_matmul_precision('medium')


NUM_CLASSES = 21
COLORS = random.Random(36).choices(list(mcolors.CSS4_COLORS.keys()), k=NUM_CLASSES)


def split_dataset(json_path: os.PathLike, split_rate: float = 0.01) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    root_dir = os.path.dirname(json_path)

    coco = COCO(json_path)
    image_ids = list(coco.imgToAnns.keys())
    random.shuffle(image_ids)

    cats = coco.dataset['categories']

    split_idx = int(split_rate * len(image_ids))
    test_ids = image_ids[:split_idx]
    train_ids = image_ids[split_idx:]

    test_imgs = coco.loadImgs(ids=test_ids)
    train_imgs = coco.loadImgs(ids=train_ids)
    
    test_anns = coco.loadAnns(coco.getAnnIds(imgIds=test_ids))
    train_anns = coco.loadAnns(coco.getAnnIds(imgIds=train_ids))

    test_coco = defaultdict(list)
    test_coco['images'] = test_imgs
    test_coco['annotations'] = test_anns
    test_coco['categories'] = cats
    with open(os.path.join(root_dir, 'val_annotations.json'), 'w') as f:
        json.dump(test_coco, f)

    train_coco = defaultdict(list)
    train_coco['images'] = train_imgs
    train_coco['annotations'] = train_anns
    train_coco['categories'] = cats
    with open(os.path.join(root_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_coco, f)


class KFashionDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dir, file_name)).convert('RGB')

        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]
        
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annot) for annot in annots], dtype=np.uint8)

        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
        }
        
        if self.transform is not None:
            image = self.transform(image)
            
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
            
        return image, target, image_id


def visualize_dataset(image_dir: os.PathLike, json_path: os.PathLike, save_dir: os.PathLike, n_images: int = 10, alpha: float = 0.5) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = KFashionDataset(
        image_dir=image_dir,
        json_path=json_path,
        transform=transforms.ToTensor()
    )

    classes = [cat['name'] for cat in dataset.coco.dataset['categories']]

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        image, target, image_id = dataset[i]
        image = image.numpy().transpose(1, 2, 0)

        plt.imshow(image)
        ax = plt.gca()

        for box, mask, category_id in zip(*target.values()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            category_id = category_id.item()

            ax.text(
                x1 + w // 2, y1 + h // 2,
                classes[category_id-1],
                c='white',
                size=10,
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS[category_id-1])],
                family='sans-serif',
                weight='semibold',
                va='center', ha='center',
            )

            contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                coords = contour.squeeze(1)
                polygon_edge = patches.Polygon(
                    (coords),
                    edgecolor=COLORS[category_id-1],
                    facecolor='none',
                    linewidth=1,
                    fill=False,
                )
                ax.add_patch(polygon_edge)
                polygon_fill = patches.Polygon(
                    (coords),
                    alpha=0.5,
                    edgecolor='none',
                    facecolor=COLORS[category_id-1],
                    fill=True
                )
                ax.add_patch(polygon_fill)

        plt.axis('off')
        file_name = dataset.coco.loadImgs(image_id)[0]['file_name']
        plt.savefig(os.path.join(save_dir, file_name), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


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
            message = 'total loss: {:>4f}, cls loss: {:>4f}, box loss: {:>4f}, obj loss: {:>4f}, rpn loss: {:>4f}, mask loss: {:>4f}  [{:>5d}/{:>5d}]'
            message = message.format(
                loss,
                loss_dict['loss_classifier'],
                loss_dict['loss_box_reg'],
                loss_dict['loss_objectness'],
                loss_dict['loss_rpn_box_reg'],
                loss_dict['loss_mask'],
                current,
                size
            )
            print(message)


class MeanAveragePrecision:
    def __init__(self, json_path: os.PathLike) -> None:
        self.coco_gt = COCO(json_path)

        self.detections = []

    def update(self, preds, image_ids):
        for p, image_id in zip(preds, image_ids):
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]
            p['boxes'] = p['boxes'].cpu().numpy()

            p['masks'] = (p['masks'].squeeze(1) >= 0.5).type(torch.uint8).cpu().numpy()

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            for b, l, s, m in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
                    'segmentation': coco_mask.encode(np.asfortranarray(m)),
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

        coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
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
    test_mask_loss = 0
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
            test_mask_loss += loss_dict['loss_mask']

            model.eval()
            preds = model(images)

            metric.update(preds, image_ids)
    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches
    test_mask_loss /= num_batches
    print(f'Test Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n Mask loss: {test_mask_loss:>8f} \n')
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

    classes = [cat['name'] for cat in testset.coco.dataset['categories']]

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, image_id = testset[i]
        image = [image.to(device)]
        pred = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        pred = {k: v.detach().cpu().numpy() for k, v in pred[0].items()}

        plt.imshow(image)
        ax = plt.gca()

        for box, category_id, score, mask in zip(*pred.values()):
            if score >= conf_thr:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                category_id = category_id.item()

                ax.text(
                    x1 + w // 2, y1 + h // 2,
                    f'{classes[category_id-1]}: {score:.2f}',
                    c='white',
                    size=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground=COLORS[category_id-1])],
                    family='sans-serif',
                    weight='semibold',
                    va='center', ha='center',
                )

                contours, _ = cv2.findContours((mask >= 0.5).astype(np.uint8).transpose(1, 2, 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    coords = contour.squeeze(1)
                    polygon_edge = patches.Polygon(
                        (coords),
                        edgecolor=COLORS[category_id-1],
                        facecolor='none',
                        linewidth=1,
                        fill=False,
                    )
                    ax.add_patch(polygon_edge)
                    polygon_fill = patches.Polygon(
                        (coords),
                        alpha=0.5,
                        edgecolor='none',
                        facecolor=COLORS[category_id-1],
                        fill=True
                    )
                    ax.add_patch(polygon_fill)

        plt.axis('off')
        file_name = testset.coco.loadImgs(image_id)[0]['file_name']
        plt.savefig(os.path.join(save_dir, file_name), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def run_pytorch(
    json_path: os.PathLike,
    image_dir: os.PathLike,
    train_json_path: os.PathLike,
    test_json_path: os.PathLike,
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
    split_dataset(json_path, split_rate=0.01)
    
    visualize_dataset(image_dir, train_json_path, save_dir='examples/k-fashion/train', alpha=0.8)
    visualize_dataset(image_dir, test_json_path, save_dir='examples/k-fashion/val', alpha=0.8)

    training_data = KFashionDataset(
        image_dir=image_dir,
        json_path=train_json_path,
        transform=transforms.ToTensor(),
    )
    test_data = KFashionDataset(
        image_dir=image_dir,
        json_path=test_json_path,
        transform=transforms.ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(json_path=test_json_path)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, device, model, optimizer)
        test(test_dataloader, device, model, metric)
    print('Done!')

    torch.save(model.state_dict(), 'k-fashion-mask-rcnn.pth')
    print('Saved PyTorch Model State to k-fashion-mask-rcnn.pth')

    model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1)
    model.load_state_dict(torch.load('k-fashion-mask-rcnn.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/k-fashion/mask-rcnn', conf_thr=0.3)


class KFashionModule(pl.LightningModule):
    def __init__(self, test_json_path, lr: Optional[float] = None):
        """_summary_

        Args:
            csv_path (_type_): _description_
            lr (_type_): _description_
        """
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1)
        self.metric = MeanAveragePrecision(test_json_path)

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
    json_path: os.PathLike,
    image_dir: os.PathLike,
    train_json_path: os.PathLike,
    test_json_path: os.PathLike,
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
    split_dataset(json_path)
    
    visualize_dataset(image_dir, train_json_path, save_dir='examples/k-fashion/train', alpha=0.8)
    visualize_dataset(image_dir, test_json_path, save_dir='examples/k-fashion/test', alpha=0.8)

    training_data = KFashionDataset(
        image_dir=image_dir,
        json_path=train_json_path,
        transform=transforms.ToTensor()
    )
    test_data = KFashionDataset(
        image_dir=image_dir,
        json_path=test_json_path,
        transform=transforms.ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, collate_fn=collate_fn)

    model = KFashionModule(test_json_path=test_json_path, lr=lr)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    trainer.save_checkpoint('k-fashion-mask-rcnn.ckpt')
    print('Saved PyTorch Lightning Model State to k-fashion-mask-rcnn.ckpt')

    model = KFashionModule.load_from_checkpoint(checkpoint_path='k-fashion-mask-rcnn.ckpt', test_json_path=test_json_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    visualize_predictions(test_data, device, model, 'examples/k-fashion/mask-rcnn-lightning', conf_thr=0.3)
