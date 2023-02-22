from hnvlib.dirty_mnist import run_pytorch


CSV_PATH = '../../data/dirty-mnist/dirty_mnist_2nd_answer.csv'
TRAIN_IMAGE_DIR = '../../data/dirty-mnist/train'
TRAIN_CSV_PATH = '../../data/dirty-mnist/train_answer.csv'
VAL_CSV_PATH = '../../data/dirty-mnist/test_answer.csv'


def main():
    run_pytorch(
        csv_path=CSV_PATH,
        image_dir=TRAIN_IMAGE_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=VAL_CSV_PATH,
        batch_size=32,
        epochs=20,
        lr=1e-4
    )


if __name__ == '__main__':
    main()
