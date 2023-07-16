from ..models.mirnetv2_builder import build_model
from pathlib import Path
from .. import io
from .. import ops
import numpy as np


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
        return self.image
    
    def _image_dim(self, images: np.ndarray):
        height,width = images.shape[2], images.shape[3]
        

    def _preprocess(self, inputs: np.ndarray):
        img_multiple_of = 4
        images_tensor = ops.to_torch(inputs)
        images_tensor = ops.pad(images_tensor, img_multiple_of)
        return images_tensor
    
    def restore(self, path: Path):
        image = self.load(path)
        h, w, d = image.shape
        
        images_pre = self._preprocess(image)
        
        images_net = self.model(images_pre)
        
        images_post = self._postprocess(images_net, h,w)
        
        return images_post
    
    def _postprocess(self, outputs, h,w):
        images_clean = ops.clamp_unpad(outputs, h, w)
        image_normalize = ops.normalize(images_clean[0])
        return image_normalize
    
    