# encoding=utf-8
from model_base import *


class MATMS_ABLA:
    pass


class CNN:
    pass


class MATMS(tnn.Module):
    def __init__(
            self,
            resolution: int = 128,
            input_dim: int = 4, embed_dim: int = 48,
            z_dim: int = 3, z_embed_dim: list = None,
            patch_size=1, depths: list = None,
            ratios: list = None, heads: int = 6,
            mlp_alpha: int = 3, window_size: int = 8,
            decoder_depth: int = 4, decoder_ch: list = None,
            drop_path_rate: float = 0.15
    ):
        super().__init__()
        z_embed_dim, depths, ratios, decoder_ch = \
            self.check_args(z_embed_dim, depths, ratios, decoder_ch,
                            resolution, embed_dim, patch_size, window_size, decoder_depth)

        self.z_mapping = MappingNet(z_dim, z_embed_dim)
        self.encoder = Encoder(resolution, input_dim, embed_dim, patch_size, depths, ratios,
                               heads, mlp_alpha, window_size, drop_path_rate)
        self.decoder = Decoder(patch_size, input_dim, embed_dim, decoder_depth, decoder_ch)

    @staticmethod
    def check_args(z_embed_dim, depths, ratios, decoder_ch,
                   resolution, embed_dim, patch_size, window_size, decoder_depth):
        assert resolution % (patch_size * window_size) == 0
        if z_embed_dim is None:
            z_embed_dim = [24, 48, embed_dim]
        if depths is None and ratios is None:
            depths = [2, 2, 2, 2, 2, 2, 2]
            ratios = [1, 0.5, 0.5, 0.5, 2, 2, 2]
        else:
            assert depths is not None and ratios is not None
            assert len(depths) == len(ratios)
        if decoder_ch is None:
            decoder_ch = [embed_dim, 32, 16, 8, 4]
        else:
            assert len(decoder_ch) == decoder_depth + 1
        return z_embed_dim, depths, ratios, decoder_ch

    def forward(self, x, z=None):
        # x: shape is (N(batch_size), 4, H(pic_size), W(pic_size))
        # channel order is inpaint_mask(0-valid, 1-invalid), prefill_ndsi, dem, tempr
        # z: shape is (N, input_dim(3)), include date / x_coord / y_coord
        identity = x

        # mp: shape is (N, C(embed_dim))
        mp_z = self.z_mapping(z)

        # transformer-based encoder
        # x: shape is (N, h(H // patch_size), w(W // patch_size), C)
        x = self.encoder(x, mp_z)

        # decoder
        # x: shape is (N, 1, H, W)
        x = self.decoder(x, identity)

        return x

    def init(self):
        self.z_mapping.init()
        self.encoder.init()
        self.decoder.init()


model_dict = {
    'CNN': CNN,
    'MAT_MS': MATMS,
    'MATMS_ABLA': MATMS_ABLA
}


def loss_func(ground_truth, prediction, cloud_mask, params=None):
    if params is None:
        params = [10, 1]
    diff = torch.square(ground_truth[:, :1] - prediction[:, :1])
    not_cloud_mask = 1 - cloud_mask
    loss = torch.sum(diff * cloud_mask * params[0]) / torch.sum(cloud_mask) + \
           torch.sum(diff * not_cloud_mask * params[1]) / torch.sum(not_cloud_mask)
    return loss
