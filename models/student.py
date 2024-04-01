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

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'


@MODELS.register_module()
class Student(BaseModel):
    # use the cat feats for da_net for residual
    def __init__(self):
        super().__init__()

        # encoder
        self.backbone = ConvNeXt(
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained', checkpoint=checkpoint_file,
                prefix='backbone.')
        )

        self.decode_head = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

        # self.load_model('checkpoints/danet.pth')

        self.reconstract = FPNHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=512,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

        self.bn_head = nn.BatchNorm2d(1)
        self.activation_head = nn.Conv2d(1024, 1, kernel_size=3, padding=1, bias=False)
        self.cls_neck = GlobalAveragePooling()
        self.fc = nn.Linear(1024, 2)

        self.load_model('/mnt/data1/qiangzeng/code/residual_reconstract/work_dirs/ssdl_casia_512x512_2/iter_4000.pth')

    def loss(self,cls_out, feats_cat, feats_residual, decoded_mask, datasample, with_cls):
        labels = []
        cls_labels = []
        for item in datasample:
            gt = item.gt_label.cuda()
            cls_gt = item.cls_label.cuda()
            labels.append(gt)
            cls_labels.append(cls_gt)
        labels = torch.stack(labels, dim=0).long()
        cls_labels = torch.stack(cls_labels, dim=0)

        decoded_mask = resize(decoded_mask, datasample[0].gt_label.shape)

        criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        loss = dict()
        loss['loss_seg'] = criterion(decoded_mask, labels)

        reconstract_real = torch.zeros(feats_residual.shape).cuda()
        inputs_real = torch.zeros(feats_cat.shape).cuda()
        for i in range(len(datasample)):
            mask = (1-datasample[i].gt_label).cuda()
            mask = resize(mask.unsqueeze(0).unsqueeze(0), feats_residual[i].shape[1:])
            mask = mask.squeeze(0)
            reconstract_real[i] = feats_residual[i] * mask
            inputs_real[i] = feats_cat[i] * mask
        recons_loss_real = F.mse_loss(reconstract_real, inputs_real)

        loss['loss_reconstract'] = recons_loss_real

        if with_cls:
            criterion = LabelSmoothLoss(label_smooth_val=0.1, loss_weight=1.0)
            loss['loss_catcls'] = criterion(cls_out, cls_labels)

        return loss

    def post_process(self, seg_out, recon, cam, residual, datasample):
        residual = resize(residual, datasample[0].label_size)
        cam = resize(cam, datasample[0].label_size)
        seg_out = resize(seg_out, datasample[0].label_size)
        seg_label = seg_out.argmax(dim=1)
        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            datasample[i].pred_label = seg_label[i]
            datasample[i].recon = recon[i]
            datasample[i].cam = cam[i]
            datasample[i].residual = residual[i]

        return datasample

    def forward(self, inputs, datasample=None, mode='loss'):
        inputs = torch.stack(inputs, dim=0)
        encoded_feats = self.backbone(inputs)

        cls_out = self.cls_neck(encoded_feats[-1])
        cls_out = self.fc(cls_out)
        cam = torch.sigmoid(self.bn_head(self.activation_head(encoded_feats[-1])))
        cam = resize(cam, encoded_feats[-1].shape[2:])

        reconstract = self.reconstract(encoded_feats)
        reconstract = resize(reconstract, inputs.shape[2:])
        residual = abs(inputs - reconstract)
        residual = residual[:,0,:,:] + residual[:,1,:,:] + residual[:,2,:,:]
        residual = residual.unsqueeze(1)

        feats = []
        for i in range(len(encoded_feats)):
            residual_map = resize(residual, encoded_feats[i].shape[2:])
            cam_map = resize(cam, encoded_feats[i].shape[2:])

            feats.append(encoded_feats[i] * residual_map + encoded_feats[i])
            feats[i] = feats[i] * cam_map + feats[i]

        decoded_mask = self.decode_head(feats)

        if mode == 'loss':
            loss = self.loss(cls_out, inputs, reconstract, decoded_mask, datasample, True)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded_mask, reconstract, cam, residual, datasample)
        else:
            return decoded_mask, cls_out, reconstract, cam, residual

    def load_model(self, file):
        ckpt = torch.load(file, map_location='cpu')

        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']

        student_state_dict = OrderedDict()

        for k, v in _state_dict.items():
            if k.startswith('student.'):
                student_state_dict[k[8:]] = v

        self.load_state_dict(student_state_dict, strict=False)

        # self.backbone.load_state_dict(backbone_state_dict, strict=False)
        # self.decode_head.load_state_dict(head_state_dict, strict=False)

        del ckpt
        torch.cuda.empty_cache()