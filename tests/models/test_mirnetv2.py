import unittest

import torch

from lumiress.models.mirnetv2_arch import MIRNetV2


class MIRNetV2TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MIRNetV2()

    def test_forward(self):
        height, width = 256, 256
        x = torch.randn(1, 3, height, width)
        out = self.model.forward(x)
        expected_shape = (1, 3, height, width)
        self.assertEqual(out.shape, expected_shape)
