# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# We follow the method of "visual commonsense r-cnn" 
# https://github.com/Wangt-CN/VC-R-CNN

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
# ------------------------------------------------------------------------------
class CausalPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CausalPredictor, self).__init__()
        num_classes = 21
        self.embedding_size = 21
        representation_size = inter_channel

        self.causal_score = nn.Linear(2*representation_size, num_classes)
        self.Wy = nn.Linear(representation_size, self.embedding_size)
        self.Wz = nn.Linear(representation_size, self.embedding_size)

        nn.init.normal_(self.causal_score.weight, std=0.01)
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        nn.init.constant_(self.causal_score.bias, 0)

        self.feature_size = representation_size
        self.dic = torch.tensor(np.load(cfg.DIC_FILE)[1:], dtype=torch.float)
        self.prior = torch.tensor(np.load(cfg.PRIOR_PROB), dtype=torch.float)

    def forward(self, x, proposals):
        device = x.get_device()
        dic_z = self.dic.to(device)
        prior = self.prior.to(device)

        box_size_list = [proposal.bbox.size(0) for proposal in proposals]
        feature_split = x.split(box_size_list)
        xzs = [self.z_dic(feature_pre_obj, dic_z, prior) for feature_pre_obj in feature_split]

        causal_logits_list = [self.causal_score(xz) for xz in xzs]

        return causal_logits_list
        
# ------------------------------------------------------------------------------
    def z_dic(self, y, dic_z, prior):
        length = y.size(0)
        if length == 1:
            print('debug')
        attention = torch.mm(self.Wy(y), self.Wz(dic_z).t()) / (self.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)
        z_hat = attention.unsqueeze(2) * dic_z.unsqueeze(0)
        z = torch.matmul(prior.unsqueeze(0), z_hat).squeeze(1)
        xz = torch.cat((y.unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*y.size(1))

        # detect if encounter nan
        if torch.isnan(xz).sum():
            print(xz)
        return xz