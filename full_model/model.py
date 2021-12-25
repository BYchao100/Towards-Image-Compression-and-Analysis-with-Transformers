import torch
import torch.nn as nn
from functools import partial

import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.models.utils import update_registered_buffers


__all__ = [
    "full_model"
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class Img2Embed(nn.Module):
    """ Encoding Image to Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        middle_chans = [128, 128, 128]
        self.proj = nn.Sequential(
            conv(in_chans, middle_chans[0]),
            nn.LeakyReLU(inplace=True),
            conv(middle_chans[0], middle_chans[1]),
            nn.LeakyReLU(inplace=True),
            conv(middle_chans[1], middle_chans[2]),
            nn.LeakyReLU(inplace=True),
            conv(middle_chans[2], embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Embed2Img(nn.Module):
    """ Decode Embedding to Image
    """
    def __init__(self, img_size=224, patch_size=16, out_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        embed_size = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        num_patches = embed_size[0] * embed_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_patches = num_patches

        middle_chans = [128, 128, 128]
        self.proj = nn.Sequential(
            deconv(embed_dim, middle_chans[0]),
            nn.LeakyReLU(inplace=True),
            deconv(middle_chans[0], middle_chans[1]),
            nn.LeakyReLU(inplace=True),
            deconv(middle_chans[1], middle_chans[2]),
            nn.LeakyReLU(inplace=True),
            deconv(middle_chans[2], out_chans)
        )

    def forward(self, x):
        B, HW, C = x.shape
        assert HW == self.num_patches, \
        f"Input embeding size ({HW}) doesn't match patches size ({self.num_patches})."
        x = self.proj(x.transpose(1, 2).reshape(B, C, self.embed_size[0], self.embed_size[1]))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class JCCTransformer(nn.Module):
    """ Joint Compression and Classification Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth

        self.patch_embed = Img2Embed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=192)
        self.chans_embed = nn.Linear(192, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Reconstruction head
        self.head_rec = Embed2Img(img_size=img_size, patch_size=patch_size, out_chans=in_chans, embed_dim=embed_dim)

        # fusion
        self.fusion0 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion1 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion2 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion3 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion = nn.Linear(embed_dim, embed_dim)

        # Compression model
        hyper_dim = 128
        self.entropy_bottleneck = EntropyBottleneck(hyper_dim)
        self.h_a = nn.Sequential(
            conv(192, hyper_dim, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(hyper_dim, hyper_dim),
            nn.LeakyReLU(inplace=True),
            conv(hyper_dim, hyper_dim, stride=1),
        )
        self.h_s = nn.Sequential(
            conv(hyper_dim, 192, stride=1),
            nn.LeakyReLU(inplace=True),
            deconv(192, 192 * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(192 * 3 //2, 192 * 2, stride=1, kernel_size=3),
        )
        self.gaussian_conditional = GaussianConditional(None)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _update_entropybottleneck(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self._update_entropybottleneck(force=force)
        return updated

    def load_state_dict(self, state_dict, pretrained=False, **kwargs):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        if not pretrained:
            update_registered_buffers(
                self.entropy_bottleneck,
                "entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
            update_registered_buffers(
                self.gaussian_conditional,
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
        super().load_state_dict(state_dict, **kwargs)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        H = W = int(N**0.5)

        # Bottleneck Compression
        y = x.transpose(1, 2).reshape(B, C, H, W)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        # Transformer
        y_hat = y_hat.flatten(2).transpose(1, 2)
        y_hat = self.chans_embed(y_hat)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        y0 = torch.cat((cls_tokens, y_hat), dim=1)
        y0 = y0 + self.pos_embed
        y0 = self.pos_drop(y0)

        y1 = self.blocks[0](y0)
        y2 = self.blocks[1](y1)
        y3 = self.blocks[2](y2)
        y4 = self.blocks[3](y3)
        y5 = self.blocks[4](y4)
        y6 = self.blocks[5](y5)
        y7 = self.blocks[6](y6)
        y8 = self.blocks[7](y7)
        y9 = self.blocks[8](y8)
        y10 = self.blocks[9](y9)
        y11 = self.blocks[10](y10)
        y12 = self.blocks[11](y11)

        y_out = self.norm(y12)

        y0 = self.fusion0(y_hat)
        y1 = self.fusion1(y1[:, 1:])
        y2 = self.fusion2(y2[:, 1:])
        y3 = self.fusion3(y3[:, 1:])

        y_rec = torch.cat((y0, y1, y2, y3), dim = 2)
        y_rec = self.fusion(y_rec)

        return (y_out[:, 0], y_rec, y_likelihoods, z_likelihoods)

    def forward(self, x):
        y_cls, y_rec, y_likelihoods, z_likelihoods = self.forward_features(x)
        cls = self.head(y_cls)
        rec = self.head_rec(y_rec)
        return (cls, rec, y_likelihoods, z_likelihoods), self.aux_loss()

    def compress(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        H = W = int(N**0.5)

        # Bottleneck Compression
        y = x.transpose(1, 2).reshape(B, C, H, W)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        # Transformer
        y_hat = y_hat.flatten(2).transpose(1, 2)
        y_hat = self.chans_embed(y_hat)

        B = y_hat.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        y0 = torch.cat((cls_tokens, y_hat), dim=1)
        y0 = y0 + self.pos_embed
        y0 = self.pos_drop(y0)

        y1 = self.blocks[0](y0)
        y2 = self.blocks[1](y1)
        y3 = self.blocks[2](y2)
        y4 = self.blocks[3](y3)
        y5 = self.blocks[4](y4)
        y6 = self.blocks[5](y5)
        y7 = self.blocks[6](y6)
        y8 = self.blocks[7](y7)
        y9 = self.blocks[8](y8)
        y10 = self.blocks[9](y9)
        y11 = self.blocks[10](y10)
        y12 = self.blocks[11](y11)

        y_out = self.norm(y12)

        y0 = self.fusion0(y_hat)
        y1 = self.fusion1(y1[:, 1:])
        y2 = self.fusion2(y2[:, 1:])
        y3 = self.fusion3(y3[:, 1:])

        y_rec = torch.cat((y0, y1, y2, y3), dim = 2)
        y_rec = self.fusion(y_rec)

        cls = self.head(y_out[:, 0])
        rec = self.head_rec(y_rec)

        return cls, rec


@register_model
def full_model(pretrained=False, **kwargs):
    model = JCCTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load("./pretrain_s/checkpoint.pth", map_location='cpu')
        model.load_state_dict(checkpoint["model"], pretrained=True, strict=False)
    return model

