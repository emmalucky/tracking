import cv2
import torch.nn as nn
import torch
import numpy as np
from torchvision.ops.boxes import nms as nms_torch

from efficientnet import EfficientNet as EffNet
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from config.config import config


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False, use_gpu=config['cuda'], crop_size=config['crop_size']):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        # del model._avg_pooling
        # del model._dropout
        for name, param in model.named_parameters():  # nn.Module有成员函数parameters()
            if "_blocks.10" in name:
                break
            param.requires_grad = False
        in_features = model._blocks_args[-1].output_filters
        out_features = 128
        model._fc = nn.Linear(in_features, out_features)
        self.model = model
        self.use_gpu = use_gpu
        self.crop_size = crop_size

    def forward_once(self, x):
        bs = x.size(0)
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        # feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        # last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # x = self.model._conv_head(x)
        x = self.model._avg_pooling(x)
        x = x.view(bs, -1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        return x.unsqueeze(0)

    def forward(self, current_image, next_image):
        current_feature = torch.zeros((1, 60, 128))
        next_feature = torch.zeros((1, 60, 128))
        if self.use_gpu:
            current_feature = current_feature.cuda()
            next_feature = next_feature.cuda()
        for i in range(current_image.shape[0]):
            # get current_feature
            feature1 = self.forward_once(current_image[i])
            current_feature = torch.cat((current_feature, feature1), 0)
            # get next_feature
            feature2 = self.forward_once(next_image[i])
            next_feature = torch.cat((next_feature, feature2), 0)
        return current_feature[1:], next_feature[1:]
