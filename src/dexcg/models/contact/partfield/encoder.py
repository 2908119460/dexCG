"""PartField point-cloud encoder used by the released DextER model."""

from dataclasses import dataclass

import torch
from easydict import EasyDict
from torch import nn

from dexcg.models.contact.partfield.modules.model_utils import VanillaMLP
from dexcg.models.contact.partfield.modules.PVCNN.encoder_pc import TriPlanePC2Encoder
from dexcg.models.contact.partfield.modules.triplane import TriplaneTransformer


@dataclass(frozen=True)
class PartFieldConfig:
    variant: str = "base"
    normalize_point_cloud: bool = True
    downsample_patch_embeddings: bool = True


def _architecture_config() -> EasyDict:
    return EasyDict(
        triplane_resolution=128,
        triplane_channels_low=128,
        triplane_channels_high=512,
        regress_2d_feat=False,
        use_pvcnnonly=True,
        use_2d_feat=False,
        pvcnn=EasyDict(
            point_encoder_type="pvcnn",
            use_point_scatter=True,
            z_triplane_channels=256,
            z_triplane_resolution=128,
            unet_cfg=EasyDict(
                depth=3,
                enabled=True,
                rolled=True,
                use_3d_aware=True,
                start_hidden_channels=32,
                use_initial_conv=False,
            ),
        ),
    )


class PartFieldEncoder(nn.Module):
    """Exact trainable PartField modules needed by DextER's point-token path."""

    point_feat_dim = 1024

    def __init__(self, config: PartFieldConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or PartFieldConfig()
        architecture = _architecture_config()
        self.triplane_resolution = architecture.triplane_resolution
        self.triplane_channels_low = architecture.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=architecture.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=architecture.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(
            input_dim=64,
            output_dim=1,
            out_activation="tanh",
            n_neurons=64,
            n_hidden_layers=6,
        )
        self.use_pvcnn = architecture.use_pvcnnonly
        self.use_2d_feat = architecture.use_2d_feat
        self.pvcnn = TriPlanePC2Encoder(
            architecture.pvcnn,
            device="cuda",
            shape_min=-1,
            shape_length=2,
            use_2d_feat=self.use_2d_feat,
        )
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

        # DextER's contact-token path consumes only the low-resolution transformer
        # tokens and the first PVCNN point features. These reconstruction-only
        # branches are present in the released checkpoint but cannot affect that
        # output, so exclude them from joint dexCG optimization and DDP reduction.
        self.logit_scale.requires_grad_(False)
        self.sdf_decoder.requires_grad_(False)
        self.triplane_transformer.upsampler.requires_grad_(False)
        self.triplane_transformer.mlp.requires_grad_(False)
        for layer in self.pvcnn.pc_encoder.encoder[1:]:
            layer.requires_grad_(False)

    def forward(
        self, point_cloud: torch.Tensor, return_point_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        xyz = point_cloud[..., :3].contiguous()
        dtype = self.pvcnn.unet_encoder.conv_final.weight.dtype
        xyz = xyz.to(dtype=dtype)
        planes = self.pvcnn(
            xyz,
            xyz,
            normalize_point_cloud=self.cfg.normalize_point_cloud,
        )
        planes, tokens = self.triplane_transformer(planes, return_tokens=True)

        if self.cfg.downsample_patch_embeddings:
            height = width = self.triplane_transformer.triplane_low_res
            batch_size, token_dim = tokens.shape[0], tokens.shape[-1]
            grid = tokens.view(batch_size, 3, height, width, token_dim)
            grid = torch.einsum("nihwd->indhw", grid).reshape(
                3 * batch_size, token_dim, height, width
            )
            grid = nn.functional.avg_pool2d(grid, kernel_size=2, stride=2)
            grid = grid.view(3, batch_size, *grid.shape[-3:])
            tokens = torch.einsum("indhw->nihwd", grid).reshape(batch_size, -1, token_dim)

        if not return_point_features:
            return tokens

        from dexcg.models.contact.partfield.modules.PVCNN.encoder_pc import sample_triplane_feat

        _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        return tokens, sample_triplane_feat(part_planes, xyz)
