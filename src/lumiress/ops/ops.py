import numpy as np
import torch
import torch.nn.functional as F


def to_torch(image: np.ndarray, device="cpu"):
    image_torch = torch.from_numpy(image).float()
    image_torch = image_torch.div(255.0).permute(2, 0, 1)
    image_torch = image_torch.unsqueeze(0).to(device)
    return image_torch


def calc_dim(value, multi=4):
    return ((value + multi) // multi) * multi


def calc_new_size(h, w, multi=4):
    nh = calc_dim(h, multi)
    nw = calc_dim(w, multi)
    return nh, nw


def calc_pad_val(new_height, height, multi=4):
    return new_height - height if height % multi != 0 else 0


def calc_pad_size(nsize, size, multi=4):
    nh, nw = nsize
    h, w = size
    ph = calc_pad_val(nh, h, multi)
    pw = calc_pad_val(nw, w, multi)
    return ph, pw


def pad(image: torch.Tensor, img_multiple_of: int = 4):
    # Calculate the new dimensions after padding
    h, w = image.shape[2], image.shape[3]
    nh, nw = calc_new_size(h, w, img_multiple_of)

    # Calculate the amount of padding needed
    pad_height, pad_width = calc_pad_size((nh, nw), (h, w), img_multiple_of)

    # Pad the image using the calculated amount of padding
    image = F.pad(image, (0, pad_width, 0, pad_height), "reflect")

    return image


def unpad(images: torch.Tensor, height, width):
    images = images[:, :, :height, :width]
    return images


def clamp_unpad(images: torch.Tensor, height: int, width: int):
    images_clamp = torch.clamp(images, 0, 1)
    images_unpadded = unpad(images_clamp, height, width)
    images_permute = images_unpadded.permute(0, 2, 3, 1).cpu().detach().numpy()
    return images_permute


def normalize(image: np.ndarray):
    img_scaled = (image - image.min()) / (image.max() - image.min())
    img_byte = (img_scaled * 255).astype("uint8")
    return img_byte


def tiling(images_tensor, model, tile=720, tile_overlap=32):
    # test the image tile by tile
    b, c, h, w = images_tensor.shape
    tile = min(tile, h, w)
    assert tile % 4 == 0, "tile size should be multiple of 4"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(images_tensor)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = images_tensor[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx : (h_idx + tile), w_idx : (w_idx + tile)].add_(out_patch)
            W[..., h_idx : (h_idx + tile), w_idx : (w_idx + tile)].add_(out_patch_mask)

    restored = E.div_(W)

    return restored
