from pathlib import Path

import numpy as np

from .. import io, ops
from ..models.mirnetv2_builder import build_model


class MIRNetv2Inference:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device

        self.model = self._build_model()

        self.filepath: Path = None
        self.dirpath: Path = None
        self.image: np.ndarray = None

    def _build_model(self):
        return build_model(name=self.name, device=self.device)

    def load(self, image_path: Path):
        self.filepath = Path(image_path)
        self.image = io.load_image(image_path)
        # return self.image

    def _preprocess(self):
        img_multiple_of = 4
        images_tensor = ops.to_torch(self.image)
        images_tensor = ops.pad(images_tensor, img_multiple_of)
        return images_tensor

    def restore(self, path: Path = None):
        if path is not None:
            self.load(path)
        images_pre = self._preprocess()
        images_net = self.model(images_pre)
        images_post = self._postprocess(images_net)

        return images_post

    def _postprocess(self, outputs):
        h, w, d = self.image.shape
        images_clean = ops.clamp_unpad(outputs, h, w)
        image_normalize = ops.normalize(images_clean[0])
        return image_normalize
