import os

import torch.distributed as dist
import torch.multiprocessing as mp

from hnvlib.coco_mini import split_dataset, visualize_dataset, COCOMiniDataset, get_train_transform, get_test_transform, run_dist, run_pytorch


ROOT_DIR = '../../data/coco2014-minival'
IMAGE_DIR = os.path.join(ROOT_DIR, 'val2014')
JSON_PATH = os.path.join(ROOT_DIR, 'instances_val2014.json')
TRAIN_JSON_PATH = os.path.join(ROOT_DIR, 'train_annotations.json')
VAL_JSON_PATH = os.path.join(ROOT_DIR, 'val_annotations.json')


def main():
    # split_dataset(json_path=JSON_PATH)
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=TRAIN_JSON_PATH,
    #     save_dir='examples/coco2014-minival/train'
    # )
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=VAL_JSON_PATH,
    #     save_dir='examples/coco2014-minival/val'
    # )
    run_pytorch(
        json_path=JSON_PATH,
        image_dir=IMAGE_DIR,
        train_json_path=TRAIN_JSON_PATH,
        test_json_path=VAL_JSON_PATH,
        batch_size=6,
        epochs=3,
        lr=0.01
    )


if __name__ == '__main__':
    # main()

    split_dataset(json_path=JSON_PATH)
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=TRAIN_JSON_PATH,
    #     save_dir='examples/coco2014-minival/train'
    # )
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=VAL_JSON_PATH,
    #     save_dir='examples/coco2014-minival/val'
    # )

    trainset = COCOMiniDataset(
        image_dir=IMAGE_DIR,
        json_path=TRAIN_JSON_PATH,
        transform=get_train_transform(),
    )
    testset = COCOMiniDataset(
        image_dir=IMAGE_DIR,
        json_path=VAL_JSON_PATH,
        transform=get_test_transform(),
    )

    dist.init_process_group('nccl')
    mp.spawn(
        run_dist,
        args=(VAL_JSON_PATH, trainset, testset, 6, 3, 0.01),
        nprocs=dist.get_world_size(),
        join=True
    )