
from lumiress.models.mirnet_v2_arch import (
    SKFF, ContextBlock, RCB,Down, DownSample, UpSample, Up, MIRNet_v2
)

import unittest
import torch


class MIRNetTestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.model = MIRNet_v2()
    
    def test_forward(self):
        height,width = 256,256
        x = torch.randn(1, 3, height, width)
        out = self.model.forward(x)
        expected_shape = (1, 3, height, width)
        self.assertEqual(out.shape, expected_shape)

