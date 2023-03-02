import os

from hnvlib.kaggle_sr import split_dataset, visualize_dataset, run_pytorch, run_pytorch_lightning


ROOT_DIR = '../../data/kaggle-sr/Data'
LR_DIR = os.path.join(ROOT_DIR, 'LR')
HR_DIR = os.path.join(ROOT_DIR, 'HR')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'train_answer.csv')
VAL_CSV_PATH = os.path.join(ROOT_DIR, 'test_answer.csv')


def main():
    # split_dataset(label_dir=HR_DIR, save_dir=ROOT_DIR)
    # visualize_dataset(
    #     lr_dir=LR_DIR,
    #     hr_dir=HR_DIR,
    #     csv_path=TRAIN_CSV_PATH,
    #     save_dir='examples/kaggle-sr/train',
    # )
    # visualize_dataset(
    #     lr_dir=LR_DIR,
    #     hr_dir=HR_DIR,
    #     csv_path=VAL_CSV_PATH,
    #     save_dir='examples/kaggle-sr/test',
    # )
    run_pytorch(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=VAL_CSV_PATH,
        batch_size=8,
        epochs=25,
        lr=1e-2
    )
    # run_pytorch_lightning(
    #     root_dir=ROOT_DIR,
    #     lr_dir=LR_DIR,
    #     hr_dir=HR_DIR,
    #     train_csv_path=TRAIN_CSV_PATH,
    #     test_csv_path=VAL_CSV_PATH,
    #     batch_size=8,
    #     epochs=15,
    # )


if __name__ == '__main__':
    main()