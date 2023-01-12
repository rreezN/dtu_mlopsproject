# -*- coding: utf-8 -*-
import logging
import pickle
from glob import glob
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('image_shape', default=224, type=click.Path())
@click.argument('norm_strat', default='model', type=click.Path())
def main(
        input_filepath: str,
        output_filepath: str,
        image_shape: int,
        norm_strat: str) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # To Tensor transformer
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    # Datasets to iterate through
    datasets = [
        "training_data", "testing_data", "validation_data", "interesting_data"
    ]

    # Perform dataset loop
    for dataset in datasets:
        # Variables in which images and labels are saved intermediately
        images = []
        labels = []
        # Iterate through all animals
        for c_idx, animal in enumerate(glob(input_filepath + f"/{dataset}/*")):
            print(f"animal: {animal}")
            # iterte through all animal images
            for file in tqdm(glob(f'{animal}/*')):
                # load image and resize it to "image_shape"
                image = Image.open(file).resize((image_shape, image_shape))
                # if loaded image is different from RGB, convert it.
                if image.mode != "RGB":
                    image = image.convert("RGB")
                # Add image to intermediate step variables
                labels.append(c_idx)
                images.append(transform_to_tensor(image))

        # Stack all images into a tensor of tensors
        images = torch.stack(images)

        # Normalisation transformer
        if norm_strat == 'model':
            # transformer based on pre-trained model, mean and std
            T = transforms.Compose([
                transforms.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])
                )
            ])
        else:
            # transformer based on caluclated means and stds.
            means = images.mean(dim=(0, 2, 3))
            stds = images.std(dim=(0, 2, 3))
            T = transforms.Compose([
                transforms.Normalize(mean=means, std=stds)
            ])
        # Transform images
        norm_images = T(images)
        torch_labels = torch.Tensor(labels)

        with open(output_filepath + f"/{dataset}.pickle", "wb") as fp:
            pickle.dump((norm_images, torch_labels), fp)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
