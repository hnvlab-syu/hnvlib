from hnvlib.mnist import run_pytorch


def main():
    run_pytorch(
        batch_size=512,
        epochs=3,
        lr=1e-2
    )


if __name__ == '__main__':
    main()
