import torch
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.backbones import ResNetV1c
from mmpretrain.models.losses import  CrossEntropyLoss
from mmseg.models.utils import resize

@MODELS.register_module()
class ResNet_Uper(BaseModel):
    def __init__(self):
        super().__init__()

        # encoder
        self.backbone = ResNetV1c(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            pretrained='open-mmlab://resnet50_v1c'
        )

        self.decode_head = UPerHead(
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

    def loss(self, decoded_mask, datasample):
        labels = []
        for item in datasample:
            gt = item.gt_label.cuda()
            labels.append(gt)
        labels = torch.stack(labels, dim=0).long()

        decoded_mask = resize(decoded_mask, datasample[0].gt_label.shape)
        loss = dict()
        criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        loss['loss'] = criterion(decoded_mask, labels)

        return loss

    def post_process(self, seg_out, datasample):
        seg_out = resize(seg_out, datasample[0].label_size)
        seg_label = seg_out.argmax(dim=1)
        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            datasample[i].pred_label = seg_label[i]
        return datasample

    def forward(self, inputs, datasample, mode):
        inputs = torch.stack(inputs, dim=0)
        encoded_feats = self.backbone(inputs)

        decoded_mask = self.decode_head(encoded_feats)

        if mode == 'loss':
            loss = self.loss(decoded_mask, datasample)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded_mask, datasample)