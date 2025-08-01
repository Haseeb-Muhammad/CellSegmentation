import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "2D_directory",
        type= str,
        required=True,
        default=""
    )

    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        default=1
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()


if __name__ == "__main__":
    main()