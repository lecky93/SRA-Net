import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmseg.models.decode_heads.da_head import DAHead
from mmpretrain.models.backbones import ConvNeXt
from mmpretrain.models.necks import GlobalAveragePooling
from mmpretrain.models.losses import LabelSmoothLoss, CrossEntropyLoss
from mmseg.models.utils import resize
from collections import OrderedDict

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'


@MODELS.register_module()
class WeaklyNet4(BaseModel):
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

        self.decode_head = DAHead(
            in_channels=1024,
            in_index=0,
            channels=512,
            pam_channels=64,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

        self.load_model('checkpoints/danet.pth')

        self.bottelneck = nn.Sequential(
            nn.Conv2d(1024 + 512 + 256 + 128, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.residual_da = DAHead(
            in_channels=1024,
            in_index=3,
            channels=512,
            pam_channels=64,
            dropout_ratio=0.1,
            num_classes=1024,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

        self.bn_head = nn.BatchNorm2d(1)
        self.activation_head = nn.Conv2d(1024, 1, kernel_size=3, padding=1, bias=False)
        self.cls_neck = GlobalAveragePooling()
        self.fc = nn.Linear(1024, 2)

    def loss(self,cls_out, feats_cat, feats_residual, decoded_mask, pam_mask, cam_mask, datasample):
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
        pam_mask = resize(pam_mask, datasample[0].gt_label.shape)
        cam_mask = resize(cam_mask, datasample[0].gt_label.shape)

        criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        loss = dict()
        loss['loss_catpam'] = criterion(pam_mask, labels)
        loss['loss_catcam'] = criterion(cam_mask, labels)
        loss['loss_catpam_catcam'] = criterion(decoded_mask, labels)

        reconstract_real = torch.zeros(feats_residual.shape).cuda()
        inputs_real = torch.zeros(feats_cat.shape).cuda()
        for i in range(len(datasample)):
            mask = (1-datasample[i].gt_label).cuda()
            mask = resize(mask.unsqueeze(0).unsqueeze(0), feats_residual[i].shape[1:])
            mask = mask.squeeze(0)
            reconstract_real[i] = feats_residual[i] * mask
            inputs_real[i] = feats_cat[i] * mask
        recons_loss = F.mse_loss(reconstract_real, inputs_real)
        loss['loss_reconstract'] = recons_loss

        criterion = LabelSmoothLoss(label_smooth_val=0.1, loss_weight=1.0)
        loss['loss_catcls'] = criterion(cls_out, cls_labels)

        return loss

    def post_process(self, seg_out, datasample):
        seg_out = resize(seg_out, datasample[0].label_size)
        seg_label = seg_out.argmax(dim=1)
        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            datasample[i].pred_label = seg_label[i]
        return datasample

    def forward(self, inputs, datasample=None, mode='loss'):
        inputs = torch.stack(inputs, dim=0)
        encoded_feats = self.backbone(inputs)

        cls_out = self.cls_neck(encoded_feats[-1])
        cls_out = self.fc(cls_out)
        cam = torch.sigmoid(self.bn_head(self.activation_head(encoded_feats[-1])))
        cam = resize(cam, encoded_feats[-1].shape[2:])

        feats_residual, _, _ = self.residual_da(encoded_feats)
        feats_residual = torch.cat([fm.unsqueeze(0) for fm in feats_residual], dim=0)
        residual = abs(encoded_feats[-1] - feats_residual)

        feats = []
        for i in range(len(encoded_feats)):
            feat = resize(encoded_feats[i], encoded_feats[-1].shape[2:])
            feats.append(feat)

        feats_cat = torch.cat(feats, dim=1)
        feats_cat = self.bottelneck(feats_cat)

        feats_final = feats_cat * residual + feats_cat
        feats_final = feats_final * cam + feats_final

        decoded_mask, pam_mask, cam_mask = self.decode_head([feats_final])

        if mode == 'loss':
            loss = self.loss(cls_out, feats_cat, feats_residual, decoded_mask, pam_mask, cam_mask, datasample)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded_mask, datasample)

    def load_model(self, file):
        ckpt = torch.load(file, map_location='cpu')

        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']

        backbone_state_dict = OrderedDict()
        head_state_dict = OrderedDict()

        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                backbone_state_dict[k[9:]] = v
            elif k.startswith('decode_head.'):
                head_state_dict[k[12:]] = v
        self.backbone.load_state_dict(backbone_state_dict, strict=False)
        self.decode_head.load_state_dict(head_state_dict, strict=False)

        del ckpt
        torch.cuda.empty_cache()

if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    model = WeaklyNet5()
    out = model(x)