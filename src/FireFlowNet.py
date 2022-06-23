import torch
from torch import nn
from basic_layers import *
from torchsummary import summary
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

class FireFlowNet(nn.Module):
    """
    FireFlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "Back to Event Basics: Self Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy", Paredes-Valles et al., 2020
    """

    def __init__(self, args):
        super().__init__()
        self._args = args

        _BASE_CHANNELS = args.base_channels

        self.E1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.E2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.R1 = build_resnet_block(_BASE_CHANNELS)
        self.E3 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.R2 = build_resnet_block(_BASE_CHANNELS)
        self.pred = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2, do_batch_norm=not self._args.no_batch_norm, ksize=1, strides=1, padding=0, activation='tanh')

    def forward(self, inputs):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        flow_dict = {}
        # forward pass
        x = inputs
        x = self.E1(x)
        x = self.E2(x)
        x = self.R1(x)
        x = self.E3(x)
        x = self.R2(x)
        flow = self.pred(x)

        flow_dict['flow3'] = flow.clone()


        return flow_dict