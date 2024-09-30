import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import ceil
from ...model.op import balanced_sampling


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation, with_bn, with_bias, l2_weight
    ):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=with_bias
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=with_bias
        )
        self.activation = F.relu if activation == "relu" else lambda x: x
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.activation(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation, with_bn, with_bias, l2_weight
    ):
        super(Conv1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=with_bias
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=with_bias
        )
        self.activation = F.relu if activation == "relu" else lambda x: x
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.activation(x)
        return x


class UNetModelBuilder(nn.Module):
    def __init__(
        self,
        downsample_factors,
        num_layers=2,
        num_units=64,
        activation="relu",
        with_bn=False,
        with_bias=True,
        skip_last_dense=False,
        num_output_channels=1,
        conv_type="2d",
        **kwargs,
    ):
        super(UNetModelBuilder, self).__init__()
        self.downsample_factors = downsample_factors
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = F.relu if activation == "relu" else lambda x: x
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.skip_last_dense = skip_last_dense
        self.num_output_channels = num_output_channels
        self.conv_type = conv_type

        conv_block_cls = ConvBlock if conv_type == "2d" else Conv1DBlock
        pool_cls = nn.MaxPool2d if conv_type == "2d" else nn.MaxPool1d

        # Encoder
        self.enc_blocks = nn.ModuleList()
        for layer_idx in range(1, num_layers + 1):
            in_channels = (
                num_units if layer_idx == 1 else num_units * 2 ** (layer_idx - 1)
            )
            out_channels = num_units * 2**layer_idx
            self.enc_blocks.append(
                conv_block_cls(
                    in_channels, out_channels, activation, with_bn, with_bias, 0.0
                )
            )
            self.enc_blocks.append(pool_cls(kernel_size=2, stride=2))

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for layer_idx in reversed(range(1, num_layers + 1)):
            in_channels = num_units * 2**layer_idx * 2
            out_channels = num_units * 2 ** (layer_idx - 1)
            self.dec_blocks.append(
                conv_block_cls(
                    in_channels, out_channels, activation, with_bn, with_bias, 0.0
                )
            )

        if conv_type == "2d":
            self.final_conv = nn.Conv2d(
                num_units, num_output_channels, kernel_size=3, padding=1
            )
            self.final_pool = nn.MaxPool2d(
                kernel_size=self.downsample_factors,
                stride=self.downsample_factors,
                ceil_mode=True,
            )
        else:
            self.final_conv = nn.Conv1d(
                num_units, num_output_channels, kernel_size=3, padding=1
            )
            self.final_pool = nn.MaxPool1d(
                kernel_size=self.downsample_factors,
                stride=self.downsample_factors,
                ceil_mode=True,
            )

    def pad_input(self, x):
        if self.conv_type == "1d":
            if x.dim() == 2:
                x = x.unsqueeze(1)

            if x.size(2) % self.downsample_factors != 0:
                x = F.pad(
                    x,
                    (0, self.downsample_factors - x.size(2) % self.downsample_factors),
                )
        elif self.conv_type == "2d":
            if x.dim() == 3:
                x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self.pad_input(x)

        skip_connections = []
        for i in range(0, len(self.enc_blocks), 2):
            enc = self.enc_blocks[i]
            pool = self.enc_blocks[i + 1]
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        skip_connections = skip_connections[::-1]

        for idx, dec in enumerate(self.dec_blocks):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip_connection = skip_connections[idx]
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)

        x = self.final_conv(x)
        if self.downsample_factors > 1:
            x = self.final_pool(x)
        x = torch.log_softmax(x.view(x.size(0), -1), dim=-1)
        return x


class MLPModelBuilder(nn.Module):
    def __init__(
        self,
        downsample_factors,
        num_layers=2,
        num_units=64,
        activation="relu",
        with_bn=False,
        p_dropout=0.0,
        **kwargs,
    ):
        super(MLPModelBuilder, self).__init__()
        self.downsample_factors = downsample_factors
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.with_bn = with_bn
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            base_num = 64
            input_units = (
                self.num_units
                if i == 0
                else max(self.num_units // (base_num**i), base_num)
            )
            output_units = max(self.num_units // (base_num ** (i + 1)), base_num)
            self.layers.append(nn.Linear(input_units, output_units))
            if self.with_bn:
                self.layers.append(nn.BatchNorm1d(output_units))
            if not np.isclose(self.p_dropout, 0):
                self.layers.append(nn.Dropout(self.p_dropout))

        self.final_layer = nn.Linear(
            output_units,
            (
                ceil(self.num_units / self.downsample_factors)
                if self.downsample_factors > 1
                else self.num_units
            ),
        )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.activation == "relu":
                    x = F.relu(x)
                # Add other activation functions here if needed
            else:
                x = layer(x)

        x = self.final_layer(x)
        x = torch.log_softmax(x, dim=-1)
        return x


class BalanceKLDivLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.kl_loss_fn = nn.KLDivLoss(reduction="none")

    def forward(self, pred, true):
        pos_mask, neg_mask = balanced_sampling(
            true, threshold=torch.finfo(true.dtype).tiny
        )

        pointwise_loss = self.kl_loss_fn(pred, true)
        pos_loss = (pointwise_loss * pos_mask.float()).sum()
        neg_loss = (pointwise_loss * neg_mask.float()).sum()
        loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
        return loss
