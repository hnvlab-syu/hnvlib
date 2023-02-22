from hnvlib.cifar10 import run_pytorch


def main():
    run_pytorch(
        batch_size=16,
        epochs=3,
        lr=1e-2
    )


if __name__ == '__main__':
    main()
