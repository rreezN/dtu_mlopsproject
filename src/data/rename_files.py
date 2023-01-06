import os
from glob import glob


def rename(path: str):

    all_files = glob(path + "/*/*/*", recursive=True)

    for file in all_files:
        os.rename(file, file.replace(" ", "_"))


if __name__ == "__main__":
    rename('../../data/raw')
