import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import json
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import autocast
from torch.cuda.amp import GradScaler
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_CLASSES = 80
COLORS = random.Random(36).choices(list(mcolors.CSS4_COLORS.keys()), k=NUM_CLASSES)


def split_dataset(json_path: os.PathLike, split_rate: float = 0.2) -> None:
    """Dirty-MNIST 데이터셋을 비율에 맞춰 train / test로 나눕니다.
    
    :param path: Dirty-MNIST 데이터셋 경로
    :type path: os.PathLike
    :param split_rate: train과 test로 데이터 나누는 비율
    :type split_rate: float
    """
    root_dir = os.path.dirname(json_path)

    coco = COCO(json_path)
    image_ids = coco.getImgIds()
    random.shuffle(image_ids)

    cat_old2new = {}
    cats = coco.dataset['categories']
    for i, new_cat_id in enumerate(range(1, len(cats)+1)):
        cat_old2new[cats[i]['id']] = new_cat_id
        cats[i]['id'] = new_cat_id

    split_idx = int(split_rate * len(image_ids))
    val_ids = image_ids[:split_idx]
    train_ids = image_ids[split_idx:]

    train_imgs = coco.loadImgs(ids=train_ids)
    val_imgs = coco.loadImgs(ids=val_ids)
    
    train_anns = coco.loadAnns(coco.getAnnIds(imgIds=train_ids, iscrowd=False))
    # print(len(train_anns))
    # train_anns = [ann for ann in train_anns if len(ann['bbox']) > 0]
    for i in range(len(train_anns)):
        train_anns[i]['category_id'] = cat_old2new[train_anns[i]['category_id']]
    # print(len(train_anns))
    val_anns = coco.loadAnns(coco.getAnnIds(imgIds=val_ids, iscrowd=False))
    # val_anns = [ann for ann in val_anns if len(ann['bbox']) > 0]
    for i in range(len(val_anns)):
        val_anns[i]['category_id'] = cat_old2new[val_anns[i]['category_id']]

    train_coco = defaultdict(list)
    train_coco['images'] = train_imgs
    train_coco['annotations'] = train_anns
    train_coco['categories'] = cats
    with open(os.path.join(root_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_coco, f)

    val_coco = defaultdict(list)
    val_coco['images'] = val_imgs
    val_coco['annotations'] = val_anns
    val_coco['categories'] = cats
    with open(os.path.join(root_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_coco, f)


class COCOMiniDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dir, file_name)).convert('RGB')
        image = np.array(image)

        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = self.coco.loadAnns(annot_ids)
        
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # if len(boxes) == 1:
        #     print(boxes)

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)

        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=target['boxes'], labels=target['labels'])
            
            image = transformed['image'] / 255.0
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            # if len(boxes) == 1:
            #     print(target['boxes'])
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            
        return image, target, image_id
    

def get_train_transform():
    return A.Compose(
        [
            A.ColorJitter(p=0.2),
            A.RGBShift(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
    )


def get_test_transform():
    return A.Compose(
        [
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
    )
    

def visualize_dataset(image_dir: os.PathLike, json_path: os.PathLike, save_dir: os.PathLike, n_images: int = 10) -> None:
    """데이터셋 샘플 bbox 그려서 시각화
    
    :param save_dir: bbox 그린 그림 저장할 폴더 경로
    :type save_dir: os.PathLike
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    dataset = COCOMiniDataset(
        image_dir=image_dir,
        json_path=json_path,
    )

    classes = [cat['name'] for cat in dataset.coco.dataset['categories']]

    indices = random.choices(range(len(dataset)), k=n_images)
    for i in tqdm(indices):
        image, target, image_id = dataset[i]

        plt.imshow(image)
        ax = plt.gca()

        for box, category_id in zip(*target.values()):
            # print(category_id)
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                w, h,
                linewidth=1,
                edgecolor=COLORS[category_id-1],
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1,
                classes[category_id-1],
                c='white',
                size=10,
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS[category_id-1])],
                family='sans-serif',
                weight='semibold',
                va='top', ha='left',
                bbox=dict(
                    boxstyle='round',
                    ec=COLORS[category_id-1],
                    fc=COLORS[category_id-1],
                )
            )

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, dataset.coco.loadImgs(image_id)[0]['file_name']), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def collate_fn(batch):
    return tuple(zip(*batch))


def train(dataloader: DataLoader, device_id: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler) -> None:
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
        images = [image for image in images]
        targets = [{k: v.to(device_id) for k, v in t.items()} for t in targets]
        # print(targets)

        with autocast(device_type='cuda', dtype=torch.float16):
            loss_dict = model(images, targets)
            # loss_dict = {k: v.mean() for k, v in loss_dict.items()}
            # print(loss_dict)
            loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(images)
            print(f'total loss: {loss:>4f}, cls loss: {loss_dict["loss_classifier"].item():>4f}, box loss: {loss_dict["loss_box_reg"].item():>4f}, obj loss: {loss_dict["loss_objectness"].item():>4f}, rpn loss: {loss_dict["loss_rpn_box_reg"].item():>4f} [{current:>5d}/{size:>5d}]')


class MeanAveragePrecision:
    def __init__(self, json_path: os.PathLike) -> None:
        self.coco_gt = COCO(json_path)

        self.detections = []
    
    def update(self, preds, image_ids):
        for p, image_id in zip(preds, image_ids):
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]
            p['boxes'] = p['boxes'].cpu().numpy()

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            for b, l, s in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': l,
                    'bbox': b.tolist(),
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


def test(dataloader: DataLoader, device_id, model: nn.Module, metric, scaler) -> None:
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
            images = [image for image in images]
            targets = [{k: v.to(device_id) for k, v in t.items()} for t in targets]

            model.train()
            with autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                # loss_dict = {k: v.mean() for k, v in loss_dict.items()}
                loss = sum(loss for loss in loss_dict.values())

            test_loss += loss
            test_cls_loss += loss_dict['loss_classifier']
            test_box_loss += loss_dict['loss_box_reg']
            test_obj_loss += loss_dict['loss_objectness']
            test_rpn_loss += loss_dict['loss_rpn_box_reg']

            model.eval()
            preds = model(images)
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

    classes = [cat['name'] for cat in testset.coco.dataset['categories']]

    model.eval()
    indices = random.Random(36).choices(range(len(testset)), k=n_images)
    for i in indices:
        image, _, image_id = testset[i]
        image = [image.to(device)]
        preds = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)

        plt.imshow(image)
        ax = plt.gca()

        preds = [{k: v.detach().cpu() for k, v in t.items()} for t in preds]
        for box, category_id, score in zip(*preds[0].values()):
            if score >= conf_thr:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                category_id = category_id.item()

                rect = patches.Rectangle(
                    (x1, y1),
                    w, h,
                    linewidth=1,
                    edgecolor=COLORS[category_id-1],
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1,
                    f'{classes[category_id-1]}: {score:.2f}',
                    c='white',
                    size=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground=COLORS[category_id-1])],
                    family='sans-serif',
                    weight='semibold',
                    va='top', ha='left',
                    bbox=dict(
                        boxstyle='round',
                        ec=COLORS[category_id],
                        fc=COLORS[category_id],
                    )
                )

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def run_dist(test_json_path, trainset, testset, batch_size, epochs, lr):
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    
    train_sampler = DistributedSampler(trainset)
    test_sampler = DistributedSampler(testset)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=1, collate_fn=collate_fn, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=1, collate_fn=collate_fn, sampler=test_sampler)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Device:', device)
    device_id = rank % torch.cuda.device_count()
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1).to(device_id)
    # model = nn.DataParallel(model)
    ddp_model = DDP(model, device_ids=[device_id], output_device=[device_id])
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    metric = MeanAveragePrecision(json_path=test_json_path)
    scaler = GradScaler()

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(trainloader, device_id, ddp_model, optimizer, scaler)
        print()
        test(testloader, device_id, ddp_model, metric, scaler)
    print('Done!')

    torch.save(model.state_dict(), 'coco2014-minival-faster-rcnn.pth')
    print('Saved PyTorch Model State to coco2014-minival-faster-rcnn.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES+1)
    model.load_state_dict(torch.load('coco2014-minival-faster-rcnn.pth'))
    model.to(device)

    visualize_predictions(testset, device, model, 'examples/coco2014-minival/faster-rcnn')


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
    split_dataset(json_path=json_path)
    
    visualize_dataset(image_dir=image_dir, json_path=train_json_path, save_dir='examples/coco2014-minival/train')
    visualize_dataset(image_dir=image_dir, json_path=test_json_path, save_dir='examples/coco2014-minival/val')

    trainset = COCOMiniDataset(
        image_dir=image_dir,
        json_path=train_json_path,
        transform=get_train_transform(),
    )
    testset = COCOMiniDataset(
        image_dir=image_dir,
        json_path=test_json_path,
        transform=get_test_transform(),
    )

    dist.init_process_group('nccl')
    run_dist(
        test_json_path=test_json_path,
        trainset=trainset,
        testset=testset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )
