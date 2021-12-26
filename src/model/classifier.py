import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from ..normalization.MPN import MPN, Triuvec
from ..normalization.svPN import svPN
from ..representation.covariance import Covariance


class Classifier(nn.Module):
    def __init__(self,num_classes=1000, input_dim=384, representationConfig={}, cls_cov=False, representationConfig_cls={}):
        super(Classifier, self).__init__()
        self.re_type = representationConfig['type']
        self.cls_cov = cls_cov
        normConfig = representationConfig['normalization']
        if self.re_type == 'second-order':
            self.representation = Covariance(**representationConfig['args'])
            if normConfig['type'] == 'svPN':
                self.normalization = svPN(**normConfig['args'])
            elif normConfig['type'] == 'MPN':
                if representationConfig['args']['cov_type'] == 'cross':
                    raise TypeError('Cross-covraiance is not supported when using MPN')
                self.normalization = MPN(**normConfig['args'])
            else:
                raise TypeError('{:} is not implemented'.format(normConfig['type']))
            self.output_dim = self.normalization.output_dim
            self.visual_fc = nn.Linear(self.output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'first-order':
            self.representation = nn.AdaptiveAvgPool1d(1)
            self.normalization = nn.Identity()
            self.visual_fc = nn.Linear(input_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if self.cls_cov:
            self.representation_cls = Covariance(**representationConfig_cls['args'])
            normConfig_cls = representationConfig_cls['normalization']
            self.normalization_cls = svPN(**normConfig_cls['args'])
            output_dim=self.normalization_cls.output_dim
            self.cls_fc = nn.Linear(output_dim, num_classes)
        else:
            self.cls_fc = nn.Linear(input_dim, num_classes)
    def _triuvec(self, x):
        return Triuvec.apply(x)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        head = x[:,0]
        if self.cls_cov:
            head = head.unsqueeze(-1).unsqueeze(-1)
            head = self.representation_cls(head)
            # head = head.reshape(x.shape[0], -1)
            head = self.normalization_cls(head)
            head = head.view(head.size(0), -1)
        head = self.cls_fc(head)
        if self.re_type is not None:
            x = x[:, 1:]
            if self.re_type == 'first-order':
                x = x.transpose(-1, -2)
            elif self.re_type == 'second-order':
                x = x.transpose(-1, -2).unsqueeze(-1)
            x = self.representation(x)
            x = self.normalization(x)
            x = x.view(x.size(0), -1)
            x = self.visual_fc(x)
            x = x + head
            return x
        else:
            return head
            
