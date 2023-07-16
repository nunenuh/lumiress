from typing import *
import os
import urllib.request
from pathlib import Path
import logging
from tqdm import tqdm
import hashlib

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


_home_dir = str(Path.home())
_base_path: str = f"{_home_dir}/.lumiress"
_base_url: str = "https://github.com/swz30/MIRNetv2/releases/download"
_weight_version = "v1.0.0"
_weight_urls = {
    _weight_version : {
        "real_denoising": {
            "url": f"{_base_url}/{_weight_version}/real_denoising.pth",
            "sha256sum": "c97fc6ce93ca041263380a00eae15087d2a70e90075a60efdd09b7b87d23d213"
        },
        "super_resolution":{
            "url": f"{_base_url}/{_weight_version}/sr_x4.pth",
            "sha256sum": "6376788b0be8991b6632185de2d91ce5d65de2362af8d4ec0e23643121eabd32"
        },
        "contrast_enhancement": {
            "url": f"{_base_url}/{_weight_version}/enhancement_fivek.pth",
            "sha256sum": "27227dc307b029a5ac18f70a6c8c209f701b2843592e165f6e1e6ddb33358e03",
        },
        "lowlight_enhancement": {
            "url": f"{_base_url}/{_weight_version}/enhancement_lol.pth",
            "sha256sum": "02d6c5ed7c34bbe8c2d8abf40f97ba28aebe8442e3695abde56636ca72e6f7c2"
        }
    }
}

_weight_base_path:str = f"{_base_path}/{_weight_version}/weights"
_weight_path = {
    "real_denoising": f"{_weight_base_path}/real_denoising.pth",
    "super_resolution": f"{_weight_base_path}/sr_x4.pth",
    "contrast_enhance": f"{_weight_base_path}/enhancement_fivek.pth",
    "lowlight_enhancement": f"{_weight_base_path}/enhancement_lol.pth"
}


class WeightDownloader:
    def __init__(self, base_path: str, weight_url: str, sha256sum: str, version: str = _weight_version, verbose: bool = False):
        self.base_path: Path = Path(base_path)
        self.weight_url: str = weight_url
        self.sha256sum: str = sha256sum
        self.version = version
        self.verbose = verbose

        self.filename: str = Path(self.weight_url).name
        self.weight_base_path: Path = Path(_weight_base_path)
        self.weight_path: Path = self.weight_base_path.joinpath(self.filename)

        self.tqdm_instance = None

    def check_and_download_weights(self):
        if not self.base_path.exists():
            os.makedirs(self.base_path)

        if not self.weight_base_path.exists():
            os.makedirs(self.weight_base_path)

        if not self.weight_path.exists():
            self.download_weights()
        else:
            if not self._checksum():
                os.remove(self.weight_path)
                self.download_weights()

    def _reporthook(self, count, block_size, total_size):
        progress_bytes = count * block_size
        self.tqdm_instance.update(progress_bytes - self.tqdm_instance.n)  

    def _calculate_sha256(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                sha256_hash.update(chunk)
        return str(sha256_hash.hexdigest())

    def _checksum(self):
        file_hash: str = self._calculate_sha256(self.weight_path)
        if self.verbose:
            logging.info(f"File checksum: {file_hash} == {self.sha256sum}")
        return self.sha256sum == file_hash

    def download_weights(self):
        if self.verbose:
            logging.info("Downloading weights from: {}".format(self.weight_url))

        self.tqdm_instance = tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading", leave=True)
        urllib.request.urlretrieve(self.weight_url, self.weight_path, reporthook=self._reporthook)

        retries = 3
        while retries > 0:
            if self._checksum():
                break

            if self.verbose:
                logging.info(f"File checksum does not match. Removing file: {self.weight_path}")

            os.remove(self.weight_path)
            self.tqdm_instance.reset()
            self.tqdm_instance.clear()
            self.tqdm_instance = tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading", leave=True)
            urllib.request.urlretrieve(self.weight_url, self.weight_path, reporthook=self._reporthook)

            retries -= 1

        if self.verbose:
            logging.info("Weights downloaded successfully")
            logging.info("Saved file to : {}".format(self.weight_path))
            logging.info("file SHA256SUM: {}".format(self._calculate_sha256(self.weight_path)))

def _download_weight(name):
    if name in _weight_urls[_weight_version].keys():
        url = _weight_urls.get(_weight_version).get(name).get("url")
        sha256sum = _weight_urls.get(_weight_version).get(name).get("sha256sum")
        wd = WeightDownloader(_base_path, url, sha256sum, verbose=True)
        wd.check_and_download_weights()
    else:
        raise ValueError(f"Weight {name} not found")
    
def download_real_denoising():
    _download_weight("real_denoising")
    
def download_super_resolution():
    _download_weight("super_resolution")
    
def download_contrast_enhancement():
    _download_weight("contrast_enhancement")
    
def download_lowlight_enhancement():
    _download_weight("lowlight_enhancement")
    
def download_all():
    download_real_denoising()
    download_super_resolution()
    download_contrast_enhancement()
    download_lowlight_enhancement()
    
