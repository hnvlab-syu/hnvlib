from hnvlib.global_wheat_detection import visualize_dataset


TRAIN_IMAGE_DIR = 'data/global-wheat-detection/train'
TRAIN_CSV_PATH = 'data/global-wheat-detection/train.csv'
TEST_IMAGE_DIR = 'data/global-wheat-detection/test'


def main():
    visualize_dataset(image_dir=TRAIN_IMAGE_DIR, csv_path=TRAIN_CSV_PATH, save_dir='examples/global-wheat-detection/train')


if __name__ == '__main__':
    main()
