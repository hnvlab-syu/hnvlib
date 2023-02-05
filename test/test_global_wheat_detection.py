from hnvlib.global_wheat_detection import visualize_dataset, run_pytorch


TRAIN_IMAGE_DIR = 'data/global-wheat-detection/train'
TRAIN_CSV_PATH = 'data/global-wheat-detection/train.csv'
TEST_IMAGE_DIR = 'data/global-wheat-detection/test'
TEST_CSV_PATH = 'data/global-wheat-detection/sample_submission.csv'


def main():
    # visualize_dataset(image_dir=TRAIN_IMAGE_DIR, csv_path=TRAIN_CSV_PATH, save_dir='examples/global-wheat-detection/train')
    run_pytorch(
        train_image_path=TRAIN_IMAGE_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        test_csv_path=TEST_CSV_PATH,
        batch_size=8,
        epochs=10,
        lr=5e-3
    )


if __name__ == '__main__':
    main()
    