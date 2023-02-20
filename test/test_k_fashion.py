import os

from hnvlib.k_fashion import split_dataset, visualize_dataset, run_pytorch


ROOT_DIR = '../../data/k-fashion'
IMAGE_DIR = os.path.join(ROOT_DIR, 'train')
JSON_PATH = os.path.join(ROOT_DIR, 'train.json')
TRAIN_JSON_PATH = os.path.join(ROOT_DIR, 'train_annotations.json')
VAL_JSON_PATH = os.path.join(ROOT_DIR, 'val_annotations.json')


def main():
    # split_dataset(json_path=JSON_PATH)
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=TRAIN_JSON_PATH,
    #     save_dir='examples/k-fashion/train',
    #     alpha=0.8
    # )
    # visualize_dataset(
    #     image_dir=IMAGE_DIR,
    #     json_path=VAL_JSON_PATH,
    #     save_dir='examples/k-fashion/val',
    #     alpha=0.8
    # )

    run_pytorch(
        json_path=JSON_PATH,
        image_dir=IMAGE_DIR,
        train_json_path=TRAIN_JSON_PATH,
        test_json_path=VAL_JSON_PATH,
        batch_size=12,
        epochs=2,
        lr=0.01
    )


if __name__ == '__main__':
    main()
