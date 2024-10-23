import torch
import torch.nn as nn
import torch.nn.functional as F

def default_unet_features():
    return nb_features = [
        [16, 32, 32, 32],            # encoder
        [32, 32, 32, 32, 32, 16, 16] # decoder
    ]

class Unet(nn.Module):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False) -> None:
        """
        Args:
            inshape: shape of the input data
            infeats: number of input channels
            nb_features: defines conv features for the unet
            nb_levels: depth of the U-net
            max_pool: down-sampling for each level
            feat_mult: multiplier for features at each level
            nb_conv_per_level: number of conv layers per level
            half_res: whether to skip the last up-sampling in the decoder
        """

        super().__init__()

        self.half_res = half_res

        if nb_features is None:
            nb_features = default_unet_features()

        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('Must provide unet nb_levels if nb_features is an integer')

            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels))

        #configure encoder
        prev_num_feats = infeats
        encoder_num_feats = [prev_num_feats]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels)

class ConvBlock(nn.Module):
    """
    Convolutional block that is used in downsampling
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act  = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.main(x)
        x = self.act(x)
        return x
