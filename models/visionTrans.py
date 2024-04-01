import torch
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.necks.multilevel_neck import MultiLevelNeck
from mmseg.models.backbones import VisionTransformer
from mmpretrain.models.losses import  CrossEntropyLoss
from mmseg.models.utils import resize

@MODELS.register_module()
class VisionTrans(BaseModel):
    def __init__(self):
        super().__init__()

        # encoder
        self.backbone = VisionTransformer(
            img_size=(512, 512),
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            final_norm=True,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic',
            pretrained='pretrain/vit_base_patch16_224.pth',
        )

        self.neck = MultiLevelNeck(
            in_channels=[768, 768, 768, 768],
            out_channels=768,
            scales=[4, 2, 1, 0.5]
        )

        self.decode_head = UPerHead(
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,)

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
        encoded_feats = self.neck(encoded_feats)
        decoded_mask = self.decode_head(encoded_feats)

        if mode == 'loss':
            loss = self.loss(decoded_mask, datasample)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded_mask, datasample)