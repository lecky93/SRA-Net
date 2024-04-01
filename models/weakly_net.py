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

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'

@MODELS.register_module()
class WeaklyNet(BaseModel):
    def __init__(self, img_length=512, hiddens=[128, 256, 512, 1024], latent_dim=512):
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
        prev_channels = hiddens[-1]
        img_length //= (4*2*2*2)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim

        # decoder
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)

        self.residual_bottleneck = nn.Sequential(
                                        nn.ConvTranspose2d(hiddens[0] + hiddens[1] + hiddens[2] + hiddens[3],
                                                           hiddens[-1],
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1),
                                        nn.BatchNorm2d(hiddens[-1]),
                                        nn.ReLU()
                                    )


        self.lateral_decoder = nn.ModuleList()

        self.lateral_decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[-1],
                                       hiddens[-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[-1]),
                    nn.ReLU()
                )
            )

        for i in range(len(hiddens) - 1, 0, -1):
            self.lateral_decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]),
                    nn.ReLU()
                )
            )

        self.reconstract = nn.Sequential(nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU()
                                         )

        self.decode_head = DAHead(
            in_channels=hiddens[-1],
            in_index=0,
            channels=512,
            pam_channels=64,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)

        # cls head
        self.bn_head = nn.BatchNorm2d(1)
        self.activation_head = nn.Conv2d(hiddens[-1], 1, kernel_size=3, padding=1, bias=False)
        self.cls_neck = GlobalAveragePooling()
        self.fc = nn.Linear(hiddens[-1], 2)


    def loss(self, inputs, reconstract, mean, logvar, decoded_mask, pam_mask, cam_mask, cls_out, datasample):
        # Hyperparameters
        kl_weight = 0.00025

        reconstract_real = torch.zeros(reconstract.shape).cuda()
        inputs_real = torch.zeros(inputs.shape).cuda()
        for i in range(len(datasample)):
            mask = (1-datasample[i].gt_label).unsqueeze(0).cuda()
            reconstract_real[i] = reconstract[i] * mask
            inputs_real[i] = inputs[i] * mask

        recons_loss = F.mse_loss(reconstract_real, inputs_real)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = dict()
        loss['loss_reconstract'] = recons_loss + kl_loss * kl_weight

        criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)

        decoded_mask = resize(decoded_mask, datasample[0].gt_label.shape)
        pam_mask = resize(pam_mask, datasample[0].gt_label.shape)
        cam_mask = resize(cam_mask, datasample[0].gt_label.shape)

        labels = []
        cls_labels = []
        for item in datasample:
            gt = item.gt_label.cuda()
            cls_gt = item.cls_label.cuda()
            labels.append(gt)
            cls_labels.append(cls_gt)
        labels = torch.stack(labels, dim=0).long()
        cls_labels = torch.stack(cls_labels, dim=0)

        loss['loss_seg'] = criterion(decoded_mask, labels) \
                           + criterion(pam_mask, labels) \
                           + criterion(cam_mask, labels)
        criterion = LabelSmoothLoss(label_smooth_val=0.1, loss_weight=1.0)
        loss['loss_cls'] = criterion(cls_out, cls_labels)

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
        feats_semantic = torch.flatten(encoded_feats[-1], 1)

        mean = self.mean_linear(feats_semantic)
        logvar = self.var_linear(feats_semantic)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))

        decoded_feats = []
        for i in range(len(self.lateral_decoder)):
            x = self.lateral_decoder[i](x)
            decoded_feats.append(x)

        residual_feats = []
        for i in range(len(encoded_feats)):
            residual_feats.append(encoded_feats[i] - resize(decoded_feats[len(decoded_feats)-1-i], encoded_feats[i].shape[2:]))

        reconstract = resize(self.reconstract(decoded_feats[-1]), inputs.shape[2:])

        cls_out = self.cls_neck(residual_feats[-1])
        cls_out = self.fc(cls_out)

        for i in range(len(residual_feats)):
            prev_shape = residual_feats[-1].shape[2:]
            residual_feats[i] = resize(
                residual_feats[i],
                size=prev_shape,
                mode='bilinear')
        residual_feats = torch.cat(residual_feats, dim=1)
        feats_cat = self.residual_bottleneck(residual_feats)


        N, C, H, W = feats_cat.size()
        cam = torch.sigmoid(self.bn_head(self.activation_head(feats_cat)))
        feats = feats_cat * cam
        decoded_mask, pam_mask, cam_mask = self.decode_head([feats])

        if mode == 'loss':
            loss = self.loss(inputs, reconstract, mean, logvar, decoded_mask, pam_mask, cam_mask, cls_out, datasample)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded_mask, datasample)