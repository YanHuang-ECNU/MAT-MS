# encoding=utf-8
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Original Written by Ze Liu
# --------------------------------------------------------
from utils import *


class PatchEmbed(tnn.Module):
    def __init__(self, resolution, input_dim, embed_dim, patch_size):
        super().__init__()
        self.proj = tnn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                               kernel_size=patch_size, stride=patch_size)
        self.norm = tnn.LayerNorm(embed_dim)
        self.scale = input_dim / embed_dim
        self.output_shape = (-1, resolution, resolution, embed_dim)

    def forward(self, x):
        # x: shape is (N, 4, H, W)
        x = self.proj(x)
        # x: shape is (N, C, h(H // patch_size, w(W // patch_size))
        x = self.norm(x.flatten(start_dim=2).transpose(1, 2)).reshape(self.output_shape)
        # x: shape is (N, h, w, C)
        return x

    def init(self):
        tnn.init.normal_(self.proj.weight)
        tnn.init.constant_(self.proj.bias, 0)


class Mlp(tnn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tnn.Linear(in_features, hidden_features)
        self.act = tnn.GELU()
        self.fc2 = tnn.Linear(hidden_features, out_features)

    def forward(self, x):
        # x: shape is (..., in_features)
        x = self.fc1(x)
        # now x shape is (..., hidden_features)
        x = self.act(x)
        x = self.fc2(x)
        # now x shape is (..., out_features)
        return x

    def init(self):
        tnn.init.normal_(self.fc1.weight, std=1)
        tnn.init.constant_(self.fc1.bias, 0)
        tnn.init.normal_(self.fc2.weight, std=1)
        tnn.init.constant_(self.fc2.bias, 0)


class WindowAttention(tnn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        embed_dim (int): Number of input channels, denoted as C
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        num_win (int): Number of Local windows, denoted as nh * nw
    """

    def __init__(self, embed_dim, num_heads, window_size, num_win):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_shape = (-1, window_size ** 2, 3, num_heads, head_dim)                 # -1 == batch_size * num_win
        self.mask_shape = (-1, num_win, num_heads, window_size ** 2, window_size ** 2)  # -1 == batch_size
        self.attn_shape = (-1, num_heads, window_size ** 2, window_size ** 2)           # -1 == batch_size * num_win

        # mlp to generate continuous relative position bias
        self.cpb_mlp = tnn.Sequential(
            tnn.Linear(2, 512, bias=True),
            tnn.ReLU(inplace=True),
            tnn.Linear(512, num_heads, bias=False))
        self.relative_coord_table, self.relative_position_index = self.build_relative_coord_data(window_size)
        self.rpb_shape = (1, window_size ** 2, window_size ** 2, num_heads)

        self.qkv = tnn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.softmax = tnn.Softmax(dim=-1)
        self.proj = tnn.Linear(in_features=embed_dim, out_features=embed_dim)

    @staticmethod
    def build_relative_coord_data(window_size, norm_scale=8):
        relative_coord = torch.arange(1 - window_size, window_size) / (window_size - 1)
        # relative_coord_table: shape is (2 * ws - 1, 2 * ws - 1, 2)
        # [..., 0] is -1 -1 -1          while [..., 1] is -1 ... 1
        #             ........                            -1 ... 1
        #             1  1   1                            -1 ... 1
        relative_coord_table = \
            torch.cat([relative_coord.reshape(-1, 1, 1).repeat(1, 2 * window_size - 1, 1),
                       relative_coord.reshape(1, -1, 1).repeat(2 * window_size - 1, 1, 1)], dim=-1)
        # relative_coord_table: x -> x * logS(1 + S * |x|)
        relative_coord_table = torch.sign(relative_coord_table) * \
                               torch.log2(torch.abs(relative_coord_table * norm_scale) + 1) / np.log2(norm_scale)
        # relative_coord_table: shape is ((2*ws-1) ** 2, 2)
        relative_coord_table = relative_coord_table.reshape(-1, 2)

        # coords: shape is (ws, ws, 2) -> (ws ** 2, 2)
        # column 0 example is   0 0 0 1 1 1 2 2 2
        # column 1 example is   0 1 2 0 1 2 0 1 2
        coords = \
            torch.cat([torch.arange(window_size).reshape(-1, 1, 1).repeat(1, window_size, 1),
                       torch.arange(window_size).reshape(1, -1, 1).repeat(window_size, 1, 1)], dim=-1).reshape(-1, 2)
        # relative_coords: shape is (ws ** 2, ws ** 2, 2)
        relative_coords = coords[:, None, :] - coords[None, :, :]
        relative_coords += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        # relative_position_index: shape is (ws ** 4, )
        relative_position_index = relative_position_index.flatten()

        return relative_coord_table.cuda(), relative_position_index.cuda()

    def forward(self, attn_windows, attn_mask):
        """
        Args:
            attn_windows: input features with shape of (N * nh * nw, ws ** 2, C)
            attn_mask: (0/-inf(actually is -100)) mask with shape of (1, nh * nw, 1, ws ** 2, ws ** 2) or None
        """
        # attn_windows = tf.normalize(attn_windows, p=2., dim=-1)
        # qkv: shape is (N * nh * nw, ws ** 2, 3C) -> (N * nh * nw, ws ** 2, 3, num_heads, head_dim)
        qkv = tf.linear(input=attn_windows, weight=self.qkv.weight, bias=self.qkv.bias).reshape(self.qkv_shape)
        # q: shape is (N * nh * nw, num_heads, ws ** 2, head_dim)
        q = tf.normalize(qkv[:, :, 0], p=2.0, dim=-1).permute(0, 2, 1, 3)
        # k: shape is (N * nh * nw, num_heads, head_dim, ws ** 2)
        k = tf.normalize(qkv[:, :, 1], p=2.0, dim=-1).permute(0, 2, 3, 1)
        # v: shape is (N * nh * nw, num_heads, ws ** 2, head_dim)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        # attn: shape is (N * nh * nw, num_heads, ws ** 2, ws ** 2)
        attn = (q @ k) * self.scale

        # rpb_table == relative_position_bias_table: shape is ((2*ws-1) ** 2, num_heads)
        rpb_table = self.cpb_mlp(self.relative_coord_table)
        # relative_position_bias: shape is (1, ws ** 2, ws ** 2, num_heads)
        relative_position_bias = rpb_table[self.relative_position_index].reshape(self.rpb_shape)
        # relative_position_bias: shape is (1, num_heads, ws ** 2, ws ** 2)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias.permute(0, 3, 1, 2))
        attn = attn + relative_position_bias

        if attn_mask is not None:
            attn = (attn.reshape(self.mask_shape) + attn_mask).reshape(self.attn_shape)
        # attn: shape is (N * nh * nw, num_heads, ws ** 2, ws ** 2)
        attn = self.softmax(attn)

        # x: shape is (N * nh * nw, num_heads, ws ** 2, head_dim) -> (N * nh * nw, ws ** 2, num_heads, head_dim)
        #    -> (N * nh * nw, ws ** 2, C)
        x = (attn @ v).transpose(1, 2).flatten(start_dim=2)
        x = self.proj(x)
        return x

    def init(self):
        tnn.init.normal_(self.qkv.weight, std=1)
        tnn.init.constant_(self.qkv.bias, 0)
        tnn.init.normal_(self.proj.weight, std=1)
        tnn.init.constant_(self.proj.bias, 0)


class SwinTransformerBlock(tnn.Module):
    r""" Swin Transformer Block.

    Args:
        input_resolution (int): Input resolution
        embed_dim (int): Number of input channels, denoted as C
        num_heads (int): Number of attention heads
        shift_size (int): Shift size for SW-MSA
        window_size (int): Local window size
        mlp_alpha (float): Ratio of mlp hidden dim to embedding dim
        drop_rate (float): Dropout rate
    """

    def __init__(self, input_resolution, embed_dim, num_heads, shift_size,
                 window_size, mlp_alpha, drop_rate):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        nh, nw = input_resolution // window_size, input_resolution // window_size
        self.partition_shape = (-1, nh, window_size, nw, window_size, embed_dim)    # (N, nh, ws, nw, ws, C)
        self.windows_shape = (-1, window_size ** 2, embed_dim)            # (N * nh * nw, ws ** 2, C)
        self.reverse_shape = (-1, nh, nw, window_size, window_size, embed_dim)      # (N, nh, nw, ws, ws, C)
        self.identity_shape = (-1, input_resolution ** 2, embed_dim)                # (N, h * w, C)

        self.attn = WindowAttention(embed_dim=embed_dim, num_heads=num_heads,
                                    window_size=window_size, num_win=nh * nw)

        self.hidden_dim = int(embed_dim * mlp_alpha)
        self.drop = tnn.Dropout(drop_rate)
        self.norm1, self.norm2 = tnn.LayerNorm(embed_dim), tnn.LayerNorm(embed_dim)
        self.ffn = tnn.Sequential(
            Mlp(in_features=embed_dim, hidden_features=self.hidden_dim),
            # tnn.Tanh(),
        )
        self.ffn_scale = 1 / mlp_alpha

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # img_mask: shape is (1, h, w, 1)
            img_mask = torch.zeros((1, input_resolution, input_resolution, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # mask_windows: shape is (1, nh, ws, nw, ws, 1) -> (1, nh, nw, ws, ws, 1)
            mask_windows = img_mask.reshape((1,) + self.partition_shape[1:-1] + (1,)).permute(0, 1, 3, 2, 4, 5)
            # mask_windows: shape is (nh * nw, ws ** 2)
            mask_windows = mask_windows.reshape((nh * nw, window_size ** 2))
            # attn_mask: shape is (nh * nw, ws ** 2, ws ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.not_equal(0) * -100 + attn_mask.eq(0) * 0
            # attn_mask: shape is (1, nh * nw, 1, ws ** 2, ws ** 2)
            attn_mask = attn_mask[None, :, None, :, :]
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # x: shape is (N, h, w, C)
        x_shape = x.shape
        # identity: shape is (N, h * w, C)
        identity = x.reshape(self.identity_shape)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # attn_windows: shape is (N, nh, ws, nw, ws, C) -> (N, nh, nw, ws, ws, C) -> (N * nh * nw, ws ** 2, C)
        attn_windows = x.reshape(self.partition_shape).permute(0, 1, 3, 2, 4, 5).reshape(self.windows_shape)
        # W-MSA/SW-MSA
        # attn_windows: shape is (N * nh * nw, ws ** 2, C)
        attn_windows = self.attn(attn_windows, self.attn_mask)
        # attn_windows: shape is (N, nh, nw, ws, ws, C) -> (N, nh, ws, nw, ws, C) -> (N, h, w, C)
        attn_windows = attn_windows.reshape(self.reverse_shape).permute(0, 1, 3, 2, 4, 5).reshape(x_shape)

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_windows = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = identity + self.drop(self.norm1(attn_windows.reshape(self.identity_shape)))
        x = self.drop(self.norm2(self.ffn(x)))     # * self.ffn_scale

        return x

    def init(self):
        self.attn.init()
        for m in self.ffn.modules():
            if isinstance(m, Mlp):
                m.init()


class PatchDownSampling(tnn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.in_res = resolution
        self.out_res = resolution // 2
        self.shp1, self.shp2 = (-1, self.out_res ** 2, dim * 4), (-1, self.out_res, self.out_res, dim)
        self.reduction = tnn.Linear(4 * dim, dim, bias=False)
        self.norm = tnn.LayerNorm(dim)
        # self.norm = tnn.Tanh()

    def forward(self, x):
        # x: shape is (N, h, w, embed_dim)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        # x: shape is (N, h // 2, w // 2, embed_dim * 4)

        x = self.reduction(x)
        # x: shape is (N, (h // 2) * (w // 2), embed_dim)
        # TODO   remove!
        x = self.norm(x)
        # x: shape is (N, h // 2, w // 2, embed_dim)
        return x

    def init(self):
        tnn.init.normal_(self.reduction.weight, std=1)


class PatchUpSampling(tnn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.in_res = resolution
        self.out_res = (self.in_res * 2, self.in_res * 2)
        # k_size = 3
        # self.conv = tnn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(k_size, k_size), stride=(1, 1),
        #                        padding=(k_size - 1) // 2, padding_mode='reflect')
        # self.scale = 1 / k_size ** 2
        self.norm = tnn.LayerNorm(dim)
        # self.act = tnn.LeakyReLU(0.1)

    def forward(self, x):
        # x: shape is (N, h, w, embed_dim)
        x = x.permute(0, 3, 1, 2)
        # x: shape is (N, embed_dim, h, w)
        x = tf.interpolate(x, size=self.out_res, mode='nearest')
        # x: shape is (N, embed_dim, 2 * h, 2 * w)
        # x = x * self.scale
        # x = self.act(self.norm(self.conv(x)))
        # x: shape is (N, embed_dim, 2 * h, 2 * w)
        x = x.permute(0, 2, 3, 1)
        # x: shape is (N, 2 * h, 2 * w, embed_dim)
        # x = self.norm(x)
        return x

    def init(self):
        pass
        # tnn.init.normal_(self.conv.weight, std=1)
        # tnn.init.constant_(self.conv.bias, 0)


class BasicLayer(tnn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        ratio (int): Ratio of resolution
        input_resolution (int): Input resolution
        dim (int): Number of input channels
        depth (int): Number of blocks
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_alpha (float): Ratio of mlp hidden dim to embedding dim
        drop_rate_lst (list[float]): Stochastic depth rate
    """

    def __init__(self, ratio, input_resolution, dim, depth, num_heads, window_size,
                 mlp_alpha, drop_rate_lst):
        super().__init__()

        self.output_resolution = int(ratio * input_resolution)
        if ratio < 1:
            self.sampler = PatchDownSampling(input_resolution, dim)
        elif ratio > 1:
            self.sampler = PatchUpSampling(input_resolution, dim)
        else:
            self.sampler = None

        self.blocks = tnn.ModuleList([
            SwinTransformerBlock(input_resolution=input_resolution, embed_dim=dim,
                                 num_heads=num_heads, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 window_size=window_size, mlp_alpha=mlp_alpha, drop_rate=drop_rate_lst[i])
            for i in range(depth)])
        self.norm = tnn.LayerNorm(dim)

    def forward(self, x):
        # x: shape is (N, h, w, C)
        identity = x
        for blk in self.blocks:
            x = blk(x)
        x = identity + x.reshape(identity.shape)
        # x: shape is (N, h', w', C)
        if self.sampler is not None:
            x = self.sampler(x)
        x = self.norm(x)
        return x

    def init(self):
        if self.sampler is not None:
            self.sampler.init()
        for blk in self.blocks:
            blk.init()


class DecoderBlock(tnn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feat_conv = tnn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, padding_mode='reflect')
        self.mask_conv = tnn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, padding_mode='reflect')
        self.norm = tnn.LayerNorm(out_channels) if out_channels > 1 else lambda a: a
        self.act = tnn.ELU()

    def forward(self, x):
        # x: shape is (N, C, h, w)
        N, _, h, w = x.shape
        x1 = self.feat_conv(x).flatten(start_dim=2).transpose(1, 2)
        x1 = self.norm(x1)
        # x1: shape is (N, h * w, C')
        x1 = self.act(x1).transpose(1, 2).reshape(N, -1, h, w)
        # x1: shape is (N, C', h, w)
        x2 = torch.sigmoid(self.mask_conv(x))
        return x1 * x2

    def init(self):
        torch.nn.init.normal_(self.feat_conv.weight)
        torch.nn.init.constant_(self.feat_conv.bias, 0)
        torch.nn.init.normal_(self.mask_conv.weight)
        torch.nn.init.constant_(self.mask_conv.bias, 0)


class MappingNet(tnn.Module):
    def __init__(self, z_dim, z_embed_dim):
        super().__init__()

        self.z_dim = z_dim
        self.layer_dim = [self.z_dim] + z_embed_dim

        input_embed = []
        for i in range(len(self.layer_dim) - 1):
            input_embed.append(tnn.Linear(self.layer_dim[i], self.layer_dim[i + 1]))
            input_embed.append(tnn.BatchNorm1d(self.layer_dim[i + 1]))
            # input_embed.append(tnn.Tanh())
        # input_embed.append(tnn.Tanh())
        self.input_embed = tnn.Sequential(*input_embed)

    def forward(self, x):
        # x: shape is (batch_size, z_dim)
        x = self.input_embed(x)
        # x: shape is (batch_size, C(embed_dim))
        return x

    def init(self):
        for m in self.input_embed.modules():
            if isinstance(m, tnn.Linear):
                tnn.init.normal_(m.weight, std=1)
                tnn.init.constant_(m.bias, 0)


class Encoder(tnn.Module):
    def __init__(self, resolution, input_dim, embed_dim, patch_size, depths, ratios,
                 heads, mlp_alpha, window_size, drop_path_rate):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depths = depths
        self.ratios = ratios
        self.heads = heads
        self.mlp_alpha = mlp_alpha
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate

        # assert resolution == 2 ** int(np.log2(resolution))

        output_res = resolution // self.patch_size
        self.patch_embed = PatchEmbed(resolution=output_res, input_dim=self.input_dim,
                                      embed_dim=self.embed_dim, patch_size=self.patch_size)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.trans_layers = tnn.ModuleList()
        for i, depth in enumerate(self.depths):
            self.trans_layers.append(
                BasicLayer(ratio=self.ratios[i], input_resolution=output_res,
                           dim=self.embed_dim, depth=depth, num_heads=self.heads,
                           window_size=self.window_size, mlp_alpha=self.mlp_alpha,
                           drop_rate_lst=dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])])
            )
            output_res = self.trans_layers[-1].output_resolution
        self.mid = (len(self.trans_layers) - 1) / 2
        # self.norm = tnn.LayerNorm(self.embed_dim)

    def forward(self, x, z):
        # x: shape is (N, 4, H, W)
        # z: shape is (N, C(embed_dim))

        # mask = x[:, 0:1]
        x = self.patch_embed(x)
        # x: shape is (N, h(H // patch_size, w(W // patch_size), C)

        skips = []
        for i, layer in enumerate(self.trans_layers):
            # x.shape is from (N, h, w, C) -> (N, h', w', C')
            x = layer(x)
            # layers should be a symmetrical size for skip connection
            if i < self.mid:
                skips.append(x)
            elif i > self.mid:
                x = x + skips[int(2 * self.mid - i)]
            else:   # i == self.mid
                x = x + z[:, None, None, :]
        # x = self.norm(x)
        return x

    def init(self):
        self.patch_embed.init()
        for layer in self.trans_layers:
            layer.init()


class Decoder(tnn.Module):
    def __init__(self, patch_size, input_dim, embed_dim, decoder_depth, decoder_ch):
        super().__init__()
        self.up_scale = patch_size
        self.identity_conv = tnn.Sequential(
            tnn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=(1, 1))
        )

        self.layers = tnn.ModuleList()
        ch_lst = decoder_ch
        for i in range(decoder_depth):
            self.layers.append(DecoderBlock(in_channels=ch_lst[i], out_channels=ch_lst[i + 1]))
        self.layers.append(tnn.Conv2d(in_channels=ch_lst[-1], out_channels=1, kernel_size=(3, 3),
                                      padding=1, padding_mode='reflect'))
        self.layers.append(tnn.Tanh())

    def forward(self, x, identity):
        # x: shape is (N, h, w, C)
        # identity: shape is (N, input_dim, H, W)
        x = x.permute(0, 3, 1, 2)
        # x: shape is (N, C, h, w)
        if self.up_scale > 1:
            x = tf.interpolate(x, size=(x.shape[-2] * self.up_scale, x.shape[-1] * self.up_scale), mode='nearest')
        # x: shape is (N, C, H, W)
        # x = torch.cat([x, self.identity_conv(identity)], dim=1)
        x = x + self.identity_conv(identity)
        # x: shape is (N, C + input_dim, H, W)

        for layer in self.layers:
            x = layer(x)
        # x: shape is (N, 1, H, W)

        return x

    def init(self):
        for m in self.identity_conv.modules():
            if isinstance(m, tnn.Conv2d):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for layer in self.layers:
            if isinstance(layer, DecoderBlock):
                layer.init()
