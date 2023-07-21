import torch
# torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import customDatasetMakers
import configparser
import dataSettings
import os
import click


def build_dataset(path_to_h5):
    config = configparser.ConfigParser()
    config.read("configs/default.cfg")
    data_filename = path_to_h5
    ip_minimum = float(config["data"]["ip_minimum"])
    ip_maximum = float(config["data"]["ip_maximum"])
    lookahead = int(config["inputs"]["lookahead"])
    lookback = int(config["inputs"]["lookback"])
    profiles = config["inputs"]["profiles"].split()
    actuators = config["inputs"]["actuators"].split()
    parameters = config["inputs"]["parameters"].split()
    space_inds = [int(key) for key in config["inputs"]["space_inds"].split()]
    # if not defined, use all data points
    if len(space_inds) == 0:
        space_inds = list(range(dataSettings.nx))
    datasetParams = {
        "lookahead": lookahead,
        "lookback": lookback,
        "space_inds": space_inds,
        "ip_minimum": ip_minimum,
        "ip_maximum": ip_maximum,
    }

    dataset = customDatasetMakers.standard_dataset(
        data_filename, profiles, actuators, parameters, **datasetParams
    )
    return dataset


@click.command()
@click.option(
    "--in_dir", required=True, type=str, help="Path to the directory containing the h5 files."
)
@click.option(
    "--out_dir",
    required=True,
    type=str,
    help="Path to the directory where the torch datasets will be saved.",
)
def build_torch_datasets(in_dir, out_dir):
    data_files = [
        "example_149057_140888.h5",
        "example_157705_149058.h5",
        "example_165399_157706.h5",
        "example_174042_165400.h5",
        "example_183223_174044.h5",
        "example_191450_183224.h5",
    ]
    full_paths = [os.path.join(in_dir, data_file) for data_file in data_files]
    print(f"Building torch datasets from the following files: {full_paths}")
    for i, full_path in enumerate(full_paths):
        print(f"Loading file number {in_dir} out of {len(data_files)}: {full_path}")

        # Load the dataset corresponding to this file.
        new_dataset = build_dataset(full_path)

        # Save the torch dataset.
        file_name_with_extension = os.path.basename(full_path)
        file_name, extension = file_name_with_extension.split(".")
        torch.save(new_dataset, os.path.join(out_dir, f"dataset_{file_name}.pt"))


if __name__ == "__main__":
    build_torch_datasets()