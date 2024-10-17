from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.ImageBranch import ImageBranch

class CRNS_NET(nn.Module):
    def __init__(self, n_classes, encoding='word_embedding'):
        super().__init__()

        self.n_classes = n_classes
        self.channels = 768
        self.device = 'cuda'
        self.backbone = ImageBranch(n_classes=self.n_classes)
        self.encoding = encoding

        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(n_classes, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(n_classes, self.channels))
            self.text_to_vision = nn.Linear(512, self.channels)
        self.class_num = n_classes

    def forward(self, x_in):
        x_in = x_in.to('cuda')

        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.organ_embedding)
            task_encoding = task_encoding
        out = self.backbone(x_in, task_encoding)
        return out



#
# net = CRNS_NET(args.num_classes).to(device)
# word_embedding = torch.load(args.word_embedding).to(device)
# net.organ_embedding.data = word_embedding.float()
