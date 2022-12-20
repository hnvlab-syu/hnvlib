from collections import defaultdict
import os
import shutil
from typing import Callable, Optional, Sequence
import random
import re

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import ToTensor, Compose
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


TRAIN_IMAGE_DIR = '../../../data/global-wheat-detection/train'
TRAIN_CSV_PATH = '../../../data/global-wheat-detection/train.csv'
TEST_IMAGE_DIR = '../../../data/global-wheat-detection/test'
TEST_CSV_PATH = '../../../data/global-wheat-detection/sample_submission.csv'


class WheatDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Optional[Sequence[Callable]] = None,
        is_test: bool = False
    ) -> None:
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
        return len(self.image_ids)

    def __getitem__(self, index: int):
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


def get_transform():
    return Compose([
        ToTensor()
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


def train(dataloader, device, model, optimizer):
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


def visualize_predictions(testset, device, model, save_dir, conf_thr: float = 0.1):
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


def run_pytorch(batch_size, epochs):
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


def main():
    run_pytorch(16, 5)


if __name__ == '__main__':
    main()
