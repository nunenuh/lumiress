import copy
import logging
from pathlib import Path

import torch

from ..io import downloader
from .mirnetv2_arch import MIRNetV2

_task_name = [
    "real_denoising",
    "super_resolution",
    "contrast_enhancement",
    "lowlight_enhancement",
]
_base_parameters = {
    "inp_channels": 3,
    "out_channels": 3,
    "n_feat": 80,
    "chan_factor": 1.5,
    "n_RRG": 4,
    "n_MRB": 2,
    "height": 3,
    "width": 2,
    "bias": False,
    "scale": 1,
}


def get_param(name):
    if name in _task_name and name == "super_resolution":
        new_parameters = copy.deepcopy(_base_parameters)
        new_parameters["scale"] = 4
        return new_parameters
    else:
        new_parameters = copy.deepcopy(_base_parameters)
        return new_parameters


def get_weight(name, device="cpu"):
    if name not in _task_name:
        raise ValueError(f"Task with name {name} does not exist")

    filepath = Path(downloader._weight_path.get(name))
    if filepath.exists() and filepath.is_file():
        weight = torch.load(str(filepath), map_location=torch.device(device))
        return weight

    logging.info(f"Weight file {name} in path {filepath} not found")
    downloader._download_weight(name)
    return get_weight(name=name, device=device)


def build_model(name, device="cpu"):
    checkpoint = get_weight(name, device=device)
    params = get_param(name)

    model = MIRNetV2(**params)
    model.load_state_dict(checkpoint["params"])
    model.eval()

    return model


def lowlight_enchancement_model(device="cpu"):
    return build_model("lowlight_enhancement", device=device)


def contrast_enhancement_model(device="cpu"):
    return build_model("contrast_enhancement", device=device)


def super_resolution_model(device="cpu"):
    return build_model("super_resolution", device=device)


def real_denoising_model(device="cpu"):
    return build_model("real_denoising", device=device)
