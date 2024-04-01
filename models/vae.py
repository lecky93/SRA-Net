import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from mmengine.registry import MODELS
from mmengine.model import BaseModel
@MODELS.register_module()
class VAE(BaseModel):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, img_length=64, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)

        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)

    def loss(self, y, y_hat, mean, logvar):
        # Hyperparameters
        kl_weight = 0.00025
        recons_loss = F.mse_loss(y_hat, y)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = dict()
        loss['loss'] = recons_loss + kl_loss * kl_weight
        return loss

    def post_process(self, pred, inputs, datasample):
        residual = abs(inputs - pred)
        seg_out = residual[:,0,:,:] + residual[:,1,:,:] + residual[:,2,:,:]
        seg_out = F.sigmoid(seg_out)

        threshold = 0.8

        for i in range(len(datasample)):
            datasample[i].pred_logits = seg_out[i]
            seg_out[seg_out >= threshold] = 1
            seg_out[seg_out < threshold] = 0
            datasample[i].pred_label = seg_out[i]
            datasample[i].recon = pred[i]
            datasample[i].residual = residual[i,0,:,:] + residual[i,1,:,:] + residual[i,2,:,:]
        return datasample

    def forward(self, inputs, datasample, mode):
        inputs = torch.stack(inputs, dim=0)
        encoded = self.encoder(inputs)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        if mode == 'loss':
            loss = self.loss(inputs, decoded, mean, logvar)
            return loss
        elif mode == 'predict':
            return self.post_process(decoded, inputs, datasample)
        else:
            return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded

    def reconstruct(self, device, dataloader):
        batch = next(iter(dataloader))
        x = batch[0:1, ...].to(device)
        output = self.forward(x)[0]
        output = output[0].detach().cpu()
        input = batch[0].detach().cpu()
        combined = torch.cat((output, input), 1)
        img = ToPILImage()(combined)
        img.save('work_dirs/tmp.jpg')

    def generate(self, device, model):
        output = self.sample(device)
        output = output[0].detach().cpu()
        img = ToPILImage()(output)
        img.save('work_dirs/tmp.jpg')