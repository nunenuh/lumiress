import unittest

import torch

from lumiress.models.mirnet_v2_arch import (MRB, RCB, RRG, SKFF, ContextBlock,
                                            Down, DownSample, Up, UpSample)


class MIRNETv2SubmoduleTestCase(unittest.TestCase):
    def test_skff(self):
        skff = SKFF(in_channels=64, height=3, reduction=8)
        inp_feats = [
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 64, 16, 16),
        ]
        output = skff.forward(inp_feats)
        self.assertEqual(output.shape, (2, 64, 16, 16))

    def test_context_block(self):
        context_block = ContextBlock(n_feat=64, bias=False)
        x = torch.randn(2, 64, 16, 16)
        output = context_block.forward(x)
        self.assertEqual(output.shape, (2, 64, 16, 16))

    def test_rcb(self):
        rcb = RCB(n_feat=64)
        x = torch.randn(1, 64, 32, 32)
        output = rcb(x)
        self.assertEqual(output.shape, x.shape)

    def test_down(self):
        down = Down(in_channels=3, chan_factor=2, bias=False)
        x = torch.randn(1, 3, 8, 8)
        out = down(x)
        self.assertEqual(out.shape, torch.Size([1, 6, 4, 4]))

    def test_downsample(self):
        downsample = DownSample(in_channels=3, scale_factor=4, chan_factor=2)
        x = torch.randn(1, 3, 16, 16)
        out = downsample(x)
        self.assertEqual(out.shape, torch.Size([1, 12, 4, 4]))

    def test_up(self):
        up = Up(in_channels=3, chan_factor=2, bias=False)
        x = torch.randn(1, 3, 4, 4)
        out = up(x)
        self.assertEqual(out.shape, torch.Size([1, 1, 8, 8]))

    def test_upsample(self):
        up = UpSample(in_channels=3, scale_factor=2, chan_factor=2, kernel_size=3)
        x = torch.randn(1, 3, 16, 16)
        out = up(x)
        self.assertEqual(out.shape, torch.Size([1, 1, 32, 32]))

    def test_mrb_forward(self):
        mrb = MRB(n_feat=64, height=256, width=256, chan_factor=2, bias=False, groups=1)
        x = torch.randn(1, 64, 256, 256)
        out = mrb(x)
        self.assertEqual(out.shape, torch.Size([1, 64, 256, 256]))

    def test_rrg_forward(self):
        rrg = RRG(
            n_feat=64,
            n_MRB=4,
            height=256,
            width=256,
            chan_factor=2,
            bias=False,
            groups=1,
        )
        x = torch.randn(1, 64, 256, 256)
        out = rrg(x)
        self.assertEqual(out.shape, torch.Size([1, 64, 256, 256]))
