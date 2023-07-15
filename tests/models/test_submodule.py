from lumiress.models.mirnet_v2_arch import (
    SKFF, ContextBlock, RCB,Down, DownSample, UpSample, Up
)

import unittest
import torch

class MIRNETv2SubmoduleTestCase(unittest.TestCase):
    def test_skff(self):
        skff = SKFF(in_channels=64, height=3, reduction=8)
        inp_feats = [
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 64, 16, 16)
        ]
        output = skff.forward(inp_feats)
        self.assertEqual(output.shape, (2, 64, 16, 16))


    def test_context_block(self):
        context_block = ContextBlock(n_feat=64, bias=False)
        x = torch.randn(2, 64, 16, 16)
        output = context_block.forward(x)
        self.assertEqual(output.shape, (2, 64, 16, 16))

    def test_rcb(self):
        rcb =  RCB(n_feat=64)
        x = torch.randn(1, 64, 32, 32)
        output = rcb(x)
        self.assertEqual(output.shape, x.shape)


    # def test_down(self):
    #     down = Down(in_channels=3, chan_factor=2, bias=False)
    #     x = torch.randn(1, 3, 8, 8)
    #     out = down(x)
    #     assert out.shape == torch.Size([1, 6, 4, 4])

    # def test_downsample(self):
    #     downsample = DownSample(in_channels=3, scale_factor=4, chan_factor=2)
    #     x = torch.randn(1, 3, 16, 16)
    #     out = downsample(x)
    #     assert out.shape == torch.Size([1, 48, 1, 1])

    # def test_up(self):
    #     up = Up(in_channels=3, chan_factor=2, bias=False)
    #     x = torch.randn(1, 3, 4, 4)
    #     out = up(x)
    #     assert out.shape == torch.Size([1, 3, 8, 8])

    # def test_upsample(self):
    #     up = UpSample(in_channels=3, chan_factor=2, bias=False)
    #     x = torch.randn(1, 3, 4, 4)
    #     out = up(x)
    #     assert out.shape == torch.Size([1, 3, 8, 8])
