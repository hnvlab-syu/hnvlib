import os
import json
import shutil
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from hnvlib.dacon_keypoint import split_dataset, visualize_dataset, run_pytorch, run_pytorch_lightning


ROOT_DIR = '../../data/dacon-keypoint'
IMAGE_DIR = os.path.join(ROOT_DIR, 'train_imgs')
CSV_PATH = os.path.join(ROOT_DIR, 'train_df.csv')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'train_answer.csv')
VAL_CSV_PATH = os.path.join(ROOT_DIR, 'test_answer.csv')

EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [6, 8], [8, 10],
    [5, 7], [7, 9], [5, 11], [11, 13], [13, 15], [6, 12],
    [12, 14], [14, 16], [5, 6], [0, 17], [5, 17], [6, 17],
    [11, 12]
]


def to_coco(csv_path: os.PathLike) -> os.PathLike:
    df = pd.read_csv(csv_path)
    
    grouped = df.groupby(by='image')
    grouped_dict = {image_id: group for image_id, group in grouped}
    
    res = defaultdict(list)

    n_id = 0
    for image_id, (file_name, group) in enumerate(grouped_dict.items()):
        with Image.open(os.path.join(IMAGE_DIR, file_name), 'r') as image:
            width, height = image.size
        res['images'].append({
            'id': image_id,
            'width': width,
            'height': height,
            'file_name': file_name,
        })

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
            'id': 0,
            'name': '_background_'
        },
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
        tmp[image_id].append(np.array(ann['keypoints']).reshape(-1, 3))

    for image_id, annots in tqdm(random.Random(36).choices(list(tmp.items()), k=n_images)):
        file_path = os.path.join(image_dir, id2fname[image_id])
        
        image = Image.open(file_path).convert('RGB')
        image = np.array(image)

        plt.imshow(image)
        ax = plt.gca()

        for keypoints in annots:
            keypoints_x = keypoints[:, 0]
            keypoints_y = keypoints[:, 1]
            for edge in EDGES:
                x = [keypoints_x[edge[0]], keypoints_x[edge[1]]]
                y = [keypoints_y[edge[0]], keypoints_y[edge[1]]]
                plt.plot(x, y, c='b', linestyle='-', linewidth=2, marker='o', markerfacecolor='g', markeredgecolor='none')

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, id2fname[image_id]), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()


def main():
    # split_dataset(csv_path=CSV_PATH)
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     csv_path=TRAIN_CSV_PATH,
    #     save_dir='examples/dacon-keypoint/train',
    # )
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     csv_path=VAL_CSV_PATH,
    #     save_dir='examples/dacon-keypoint/test',
    # )

    # json_path = to_coco(VAL_CSV_PATH)
    # visualize_coco(json_path, IMAGE_DIR, 'examples/dacon-keypoint/coco')

    run_pytorch(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=VAL_CSV_PATH,
        batch_size=8,
        epochs=3,
        lr=1e-2
    )
    # run_pytorch_lightning(
    #     csv_path=CSV_PATH,
    #     image_dir=IMAGE_DIR,
    #     train_csv_path=TRAIN_CSV_PATH,
    #     test_csv_path=VAL_CSV_PATH,
    #     batch_size=8,
    #     epochs=3,
    #     lr=1e-2
    # )


if __name__ == '__main__':
    main()