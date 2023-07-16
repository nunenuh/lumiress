import unittest
from pathlib import Path
import numpy as np
from lumiress.models.mirnetv2_builder import build_model
from lumiress import io
from lumiress import ops
from lumiress.infer.mirnetv2_infer import MIRNetv2Inference
import os

class TestMIRNetv2Inference(unittest.TestCase):
    def setUp(self):
        path = os.getcwd()
        self.image_path = Path(path).joinpath("assets/tests_images/degraded/1.png")
        self.inference_obj = MIRNetv2Inference(name="real_denoising", device="cpu")
        self.inference_obj.load(self.image_path)

    def test_preprocess(self):
        expected_shape = (1, 3, 496, 464) # adjust the expected shape based on your implementation
        images_tensor = self.inference_obj._preprocess()
        self.assertEqual(images_tensor.shape, expected_shape)

    # def test_restore(self):
    #     expected_result_shape = (256, 256, 3) # adjust the expected shape based on your implementation
    #     images_post = self.inference_obj.restore()
    #     self.assertEqual(images_post.shape, expected_result_shape)


    def test_postprocess(self):
        expected_result_shape = (496, 464, 3) # adjust the expected shape based on your implementation
        images_tensor = self.inference_obj._preprocess()
        image_normalize = self.inference_obj._postprocess(images_tensor)
        self.assertEqual(image_normalize.shape, expected_result_shape)

if __name__ == '__main__':
    unittest.main()
