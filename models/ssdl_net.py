import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmseg.models.decode_heads.da_head import DAHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.fpn_head import FPNHead
from mmpretrain.models.backbones import ConvNeXt
from mmpretrain.models.necks import GlobalAveragePooling
from mmpretrain.models.losses import LabelSmoothLoss, CrossEntropyLoss
from mmseg.models.utils import resize
from collections import OrderedDict
from .weakly_net7 import WeaklyNet7

import numpy as np

@MODELS.register_module()
class SSDLNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.teacher = WeaklyNet7()
        self.student = WeaklyNet7()
        # self.load_teacher('checkpoints/teacher.pth')
        self.step = 0

    def forward(self, inputs, datasample=None, mode='loss'):
        with torch.no_grad():
            t_pseudo_label, _, t_reconstracts, _, _ = self.teacher(inputs, datasample, mode='tensor')

        s_out, s_cls_label, s_reconstracts, cam, residual = self.student(inputs, datasample, mode='tensor')

        inputs = torch.stack(inputs, dim=0)
        if mode == 'loss':
            self.update_ema_variables(0.99)
            loss = self.loss(t_pseudo_label, s_out, s_cls_label, s_reconstracts, t_reconstracts, inputs, datasample)
            return loss
        elif mode == 'predict':
            return self.post_process(s_out, s_reconstracts, cam, residual, datasample)

    def loss(self, t_pseudo_label, s_out, s_cls_label, s_reconstracts, t_reconstracts, inputs, datasample):
        labels = []
        cls_labels = []
        for item in datasample:
            gt = item.gt_label.cuda()
            cls_gt = item.cls_label.cuda()
            labels.append(gt)
            cls_labels.append(cls_gt)
        labels = torch.stack(labels, dim=0).long()
        cls_labels = torch.stack(cls_labels, dim=0)


        count = len(datasample)
        loss = dict()
        criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        index = 0
        loss_seg = 0
        for item in datasample:
            if item.cls_label == 1 and item.gt_label.max() == 0:
                # use pseudo
                tmp = F.mse_loss(s_out[index].unsqueeze(0), t_pseudo_label[index].unsqueeze(0))
            else:
                decoded_mask = resize(s_out[index].unsqueeze(0), labels[index].shape)
                tmp = criterion(decoded_mask, labels[index].unsqueeze(0))
            loss_seg += tmp
            index+=1

        loss['loss_seg'] = loss_seg / count

        reconstract_real = torch.zeros(s_reconstracts.shape).cuda()
        inputs_real = torch.zeros(inputs.shape).cuda()
        for i in range(len(datasample)):
            mask = (1-datasample[i].gt_label).cuda()
            mask = resize(mask.unsqueeze(0).unsqueeze(0), s_reconstracts[i].shape[1:])
            mask = mask.squeeze(0)
            reconstract_real[i] = s_reconstracts[i] * mask
            inputs_real[i] = inputs[i] * mask
        recons_loss_real = F.mse_loss(reconstract_real, inputs_real)
        loss['loss_reconstract'] = recons_loss_real

        loss['loss_streconstract'] = F.mse_loss(s_reconstracts, t_reconstracts)

        criterion = LabelSmoothLoss(label_smooth_val=0.1, loss_weight=1.0)
        loss['loss_catcls'] = criterion(s_cls_label, cls_labels)

        return loss

    def post_process(self, seg_out, recon, cam, residual, datasample):
        residual = resize(residual, datasample[0].label_size)
        cam = resize(cam, datasample[0].label_size)
        seg_out = F.softmax(seg_out, dim=1)
        seg_out = resize(seg_out, datasample[0].label_size)
        seg_label = seg_out.argmax(dim=1)
        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            datasample[i].pred_label = seg_label[i]
            datasample[i].recon = recon[i]
            datasample[i].cam = cam[i]
            datasample[i].residual = residual[i]
        return datasample


    def update_ema_variables(self, alpha):
        self.step += 0
        if self.step >= 2000:
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.data = alpha * t.data + (1-alpha) * s.data

    def load_teacher(self, file):
        ckpt = torch.load(file, map_location='cpu')

        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']

        self.teacher.load_state_dict(_state_dict)

        del ckpt
        torch.cuda.empty_cache()


import datetime
if __name__ == '__main__':
    model = SSDLNet().cuda()
    x = torch.randn(1,3,512,512).cuda()
    start = datetime.datetime.now()
    y = model(x)
    end = datetime.datetime.now()
    print('time', end-start)
