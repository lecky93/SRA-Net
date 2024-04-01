import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.backbones.swin import SwinBlockSequence
from mmpretrain.models.backbones import ConvNeXt, ResNetV1c, VGG
from collections import OrderedDict
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.losses import CrossEntropyLoss, accuracy
from losses import AdaptiveCELoss

from mmcv.cnn import ConvModule, Scale
from mmcv.cnn import build_norm_layer
from typing import Optional
from mmpretrain.models.selfsup import MAEViT
from mmpretrain.models.necks import MAEPretrainDecoder
# from mmpretrain.models.heads import MAEPretrainHead

class PixelReconstructionLoss(nn.Module):
    """Loss for the reconstruction of pixel in Masked Image Modeling.

    This module measures the distance between the target image and the
    reconstructed image and compute the loss to optimize the model. Currently,
    This module only provides L1 and L2 loss to penalize the reconstructed
    error. In addition, a mask can be passed in the ``forward`` function to
    only apply loss on visible region, like that in MAE.

    Args:
        criterion (str): The loss the penalize the reconstructed error.
            Currently, only supports L1 and L2 loss
        channel (int, optional): The number of channels to average the
            reconstruction loss. If not None, the reconstruction loss
            will be divided by the channel. Defaults to None.
    """

    def __init__(self, criterion: str, channel: Optional[int] = None) -> None:
        super().__init__()

        if criterion == 'L1':
            self.penalty = torch.nn.L1Loss(reduction='none')
        elif criterion == 'L2':
            self.penalty = torch.nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f'Currently, PixelReconstructionLoss \
            only supports L1 and L2 loss, but get {criterion}')

        self.channel = channel if channel is not None else 1

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function to compute the reconstrction loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        loss = self.penalty(pred, target)

        # if the dim of the loss is 3, take the average of the loss
        # along the last dim
        if len(loss.shape) == 3:
            loss = loss.mean(dim=-1)

        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum() / self.channel

        return loss


@MODELS.register_module()
class AuMAE(BaseModel):
    def __init__(self, arch='b', patch_size=16, mask_ratio=0.75):
        super(AuMAE, self).__init__()
        self.backbone = MAEViT(
            arch=arch,
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )

        self.neck = MAEPretrainDecoder(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.
        )

        self.backbone.init_weights()
        self.neck.init_weights()

        self.norm_pix = True
        self.patch_size = patch_size
        self.loss_module = PixelReconstructionLoss(criterion='L2')

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, 3, H, W)`.

        Returns:
            torch.Tensor: Patchified images. The shape is
            :math:`(B, L, \text{patch_size}^2 \times 3)`.
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_size}^2 \times 3)`.

        Returns:
            torch.Tensor: The shape is :math:`(B, 3, H, W)`.
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.

        Args:
            target (torch.Tensor): Image with the shape of B x 3 x H x W

        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def loss(self, pred, target, mask):
        target = self.construct_target(target)
        loss = dict()
        loss['loss'] = self.loss_module(pred, target, mask)

        return loss

    def post_process(self, pred, inputs, datasample):
        imgs = self.unpatchify(pred)
        residual = abs(inputs - imgs)
        seg_out = residual[:,0,:,:] + residual[:,1,:,:] + residual[:,2,:,:]
        seg_out = F.sigmoid(seg_out)

        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            seg_out[seg_out >= 0.92] = 1
            seg_out[seg_out < 0.92] = 0
            datasample[i].pred_label = seg_out[i]
        return datasample

    def forward(self, inputs, datasample, mode):
        inputs = torch.stack(inputs, dim=0)
        x, mask, ids_restore = self.backbone(inputs)
        pred = self.neck(x, ids_restore)

        if mode == 'loss':
            loss = self.loss(pred, inputs, mask)
            return loss
        elif mode == 'predict':
            return self.post_process(pred, inputs, datasample)