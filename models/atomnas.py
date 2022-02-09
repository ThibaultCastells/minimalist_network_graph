"""Searched network builder."""
import warnings
from torch import nn

from models.mobilenet_base import ConvBNReLU
from models.mobilenet_base import get_active_fn
from models.mobilenet_base import get_block
from models.mobilenet_base import InvertedResidualChannelsFused
from models.mobilenet_base import _get_named_block_list

__all__ = ['MobileNetSearched']


class MobileNetSearched(nn.Module):
    """MobileNetV2-like network."""

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_channel=32,
                 last_channel=1280,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 dropout_ratio=0.2,
                 se_ratio=None,
                 batch_norm_momentum=0.1,
                 batch_norm_epsilon=1e-5,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 round_nearest=8):
        """Build the network.
        Args:
            num_classes (int): Number of classes
            input_size (int): Input resolution.
            input_channel (int): Number of channels for stem convolution.
            last_channel (int): Number of channels for stem convolution.
            width_mult (float): Width multiplier - adjusts number of channels in
                each layer by this amount
            inverted_residual_setting (list): A list of
                [expand ratio, output channel, num repeat,
                stride of first block, A list of kernel sizes].
            dropout_ratio (float): Dropout ratio for linear classifier.
            se_ratio (float): SE ratio, for `InvertedResidualChannelsFused`
                block only.
            batch_norm_momentum (float): Momentum for batch normalization.
            batch_norm_epsilon (float): Epsilon for batch normalization.
            active_fn (str): Specify which activation function to use.
            block (str): Specify which MobilenetV2 block implementation to use.
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number Set to 1 to turn off rounding.
        """
        super(MobileNetSearched, self).__init__()
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        if width_mult != 1.0:
            raise ValueError('Searched model should have width 1')

        self.input_size = input_size
        self.input_channel = input_channel
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.inverted_residual_setting = inverted_residual_setting
        self.active_fn = active_fn
        self.block = block
        self.batch_norm_kwargs = batch_norm_kwargs

        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 6:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 6-element list, got {}".format(inverted_residual_setting))
        if input_size % 32 != 0:
            raise ValueError('Input size must divide 32')
        for name, channel in [['Input', input_channel], ['Last', last_channel]]:
            if (channel * width_mult) % round_nearest:
                warnings.warn('{} channel could not divide {}'.format(
                    name, round_nearest))
        active_fn = get_active_fn(active_fn)
        block = get_block(block)
        _extra_kwargs = {}
        if se_ratio is not None:
            if issubclass(block, InvertedResidualChannelsFused):
                _extra_kwargs['se_ratio'] = se_ratio
            else:
                raise NotImplementedError(
                    'SE module not supported for block: {}'.format(block))

        # building first layer
        features = [
            ConvBNReLU(3,
                       input_channel,
                       stride=2,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn)
        ]
        # building inverted residual blocks
        for c, n, s, ks, hiddens, expand in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          hiddens,
                          ks,
                          expand,
                          active_fn=active_fn,
                          batch_norm_kwargs=batch_norm_kwargs,
                          **_extra_kwargs))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel,
                       last_channel,
                       kernel_size=1,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn))
        avg_pool_size = input_size // 32
        features.append(nn.AvgPool2d(avg_pool_size))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, num_classes),
        )

    def get_named_block_list(self):
        """Get `{name: module}` dictionary for all inverted residual blocks."""
        return _get_named_block_list(self)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x


def atomnas_a():
    inverted_residual_setting = [[16, 1, 1, [3], [16], False],
                                  [24, 1, 2, [3, 5, 7], [16, 12, 9], True],
                                  [24, 1, 1, [3, 5, 7], [19, 2, 1], True],
                                  [24, 1, 1, [3, 5, 7], [9, 7, 1], True],
                                  [24, 1, 1, [3, 5], [14, 1], True],
                                  [40, 1, 2, [3, 5, 7], [46, 49, 50], True],
                                  [40, 1, 1, [3, 5, 7], [27, 19, 14], True],
                                  [40, 1, 1, [3, 5, 7], [22, 11, 3], True],
                                  [40, 1, 1, [3, 5, 7], [30, 14, 3], True],
                                  [80, 1, 2, [3, 5, 7], [127, 130, 111], True],
                                  [80, 1, 1, [3, 5, 7], [33, 25, 45], True],
                                  [80, 1, 1, [3, 5, 7], [38, 12, 34], True],
                                  [80, 1, 1, [3, 5, 7], [53, 23, 19], True],
                                  [96, 1, 1, [3, 5, 7], [344, 196, 168], True],
                                  [96, 1, 1, [3, 5, 7], [44, 18, 27], True],
                                  [96, 1, 1, [3, 5, 7], [37, 29, 18], True],
                                  [96, 1, 1, [3, 5, 7], [52, 21, 24], True],
                                  [192, 1, 2, [3, 5, 7], [381, 364, 342], True],
                                  [192, 1, 1, [3, 5, 7], [123, 76, 162], True],
                                  [192, 1, 1, [3, 5, 7], [179, 107, 212], True],
                                  [192, 1, 1, [3, 5, 7], [238, 111, 171], True],
                                  [320, 1, 1, [3, 5, 7], [654, 495, 546], True]]
    return MobileNetSearched(input_channel = 16, active_fn = 'nn.ReLU', inverted_residual_setting=inverted_residual_setting)