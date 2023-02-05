import os
import json
import shutil
import random
from collections import defaultdict
from ast import literal_eval

import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch

from hnvlib.global_wheat_detection import split_dataset, visualize_dataset, run_pytorch, run_pytorch_lightning


CSV_PATH = '../../data/global-wheat-detection/train.csv'
TRAIN_IMAGE_DIR = '../../data/global-wheat-detection/train'
TRAIN_CSV_PATH = '../../data/global-wheat-detection/train_answer.csv'
VAL_CSV_PATH = '../../data/global-wheat-detection/test_answer.csv'


def advanced_configure_optimizers(self):
    params = [p for p in self.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    return [optimizer], [scheduler]


def to_coco(csv_path: os.PathLike) -> os.PathLike:
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

    res['categories'].append([{'id': 0, 'name': '_background_'}, {'id': 1, 'name': 'wheat'}])
        
    root_dir = os.path.split(csv_path)[0]
    save_path = os.path.join(root_dir, 'coco_annotations.json')
    with open(save_path, 'w') as f:
        json.dump(res, f)

    return save_path


def visualize_coco(json_path: os.PathLike, image_dir: os.PathLike, save_dir: os.PathLike, n_images: int = 10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    id2fname = {i['id']: i['file_name'] for i in json_file['images']}
    
    tmp = defaultdict(list)
    for ann in tqdm(json_file['annotations']):
        image_id = ann['image_id']
        tmp[image_id].append(ann['bbox'])

    for image_id, annots in tqdm(random.Random(36).choices(list(tmp.items()), k=n_images)):
        file_path = os.path.join(image_dir, id2fname[image_id])
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        ax = plt.gca()

        for x1, y1, w, h in annots:
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
        plt.savefig(os.path.join(save_dir, id2fname[image_id]), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def main():
    # split_dataset(csv_path=CSV_PATH)
    # visualize_dataset(image_dir=TRAIN_IMAGE_DIR, csv_path=TRAIN_CSV_PATH, save_dir='examples/global-wheat-detection/train')
    # visualize_dataset(image_dir=TRAIN_IMAGE_DIR, csv_path=VAL_CSV_PATH, save_dir='examples/global-wheat-detection/test')
    
    # json_path = to_coco(VAL_CSV_PATH)
    # visualize_coco(json_path, TRAIN_IMAGE_DIR, 'examples/global-wheat-detection/coco')
    
    # run_pytorch(
    #     csv_path=CSV_PATH,
    #     train_image_dir=TRAIN_IMAGE_DIR,
    #     train_csv_path=TRAIN_CSV_PATH,
    #     test_csv_path=VAL_CSV_PATH,
    #     batch_size=16,
    #     epochs=2,
    #     lr=1e-3
    # )
    run_pytorch_lightning(
        csv_path=CSV_PATH,
        train_image_dir=TRAIN_IMAGE_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=VAL_CSV_PATH,
        batch_size=16,
        epochs=2,
        lr=1e-3
    )


if __name__ == '__main__':
    main()
