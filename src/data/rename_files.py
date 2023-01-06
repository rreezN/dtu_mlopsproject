from glob import glob
import os

def rename(path):

    all_files = glob(path + "/*/*/*", recursive=True)

    for file in all_files:
        os.rename(file, file.replace(" ", "_"))

if __name__ == "__main__":
    rename('../../data/raw')