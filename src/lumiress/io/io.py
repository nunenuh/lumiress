from pathlib import Path

import cv2
import numpy as np

_extension_allowed = ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG", "bmp", "BMP"]


def load_image(filepath: str) -> np.ndarray:
    filepath: Path = Path(filepath)
    file_ext = filepath.suffix.replace(".", "")
    if file_ext in _extension_allowed:
        if filepath.exists():
            img = cv2.imread(str(filepath))
            norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return norm
        else:
            raise ValueError(f"File {filepath} not found")
    else:
        raise ValueError(
            f"File extension {filepath.suffix} are not allowed, extension that allowed are {_extension_allowed}"
        )


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
