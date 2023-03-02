import os
from typing import Tuple, Optional, Sequence, Callable

import numpy as np

from hnvlib.pascal_voc_2012 import split_dataset, visualize_dataset, run_pytorch, run_pytorch_lightning


ROOT_DIR = '../../data/pascal-voc-2012/VOCdevkit/VOC2012'
IMAGE_DIR = os.path.join(ROOT_DIR, 'JPEGImages')
LABEL_DIR = os.path.join(ROOT_DIR, 'SegmentationClass')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'train_answer.csv')
VAL_CSV_PATH = os.path.join(ROOT_DIR, 'test_answer.csv')


def rle_decode(mask_rle: str, shape: Tuple[int]) -> np.ndarray:
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order='F')


def main():
    # split_dataset(label_dir=LABEL_DIR, save_dir=ROOT_DIR)
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     label_dir=LABEL_DIR,
    #     csv_path=TRAIN_CSV_PATH,
    #     save_dir='examples/pascal-voc-2012/train',
    #     alpha=0.8
    # )
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     label_dir=LABEL_DIR,
    #     csv_path=VAL_CSV_PATH,
    #     save_dir='examples/pascal-voc-2012/val',
    #     alpha=0.8
    # )
    run_pytorch(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=VAL_CSV_PATH,
        batch_size=8,
        epochs=30,
        lr=1e-2,
        size=(500, 500)
    )
    # run_pytorch_lightning(
    #     root_dir=ROOT_DIR,
    #     image_dir=IMAGE_DIR,
    #     label_dir=LABEL_DIR,
    #     train_csv_path=TRAIN_CSV_PATH,
    #     test_csv_path=VAL_CSV_PATH,
    #     batch_size=8,
    #     epochs=30,
    #     size=(500, 500)
    # )


if __name__ == '__main__':
    main()
