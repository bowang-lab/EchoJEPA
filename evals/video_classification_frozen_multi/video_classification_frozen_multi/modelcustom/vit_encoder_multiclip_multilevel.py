# evals/video_classification_frozen_multi/modelcustom/vit_encoder_multiclip_multilevel.py
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    out_layers = wrapper_kwargs.get("out_layers")

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, out_layers=out_layers, **enc_kwargs
    )

    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    print(model)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class ClipAggregation(nn.Module):
    """
    Process each clip indepdnently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        out_layers=None,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        num_clips = len(x)              # = 4 (total view×clip combinations)
        num_views_per_clip = len(x[0])  # = 1 (spatial augmentations)
        B, C, F, H, W = x[0][0].size()
    
        # Concatenate all views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
    
        outputs = self.model(x)
        outputs = torch.cat(outputs, dim=1)  # Multi-layer concat: [B*num_clips*num_views, N, D]
    
        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size
            S = N // T
    
            eff_B = B * num_views_per_clip
            
            # Return L = num_clips separate slot tensors (one per view×clip)
            all_outputs = []
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                # If multiple spatial views, average them; otherwise just take the single view
                if num_views_per_clip > 1:
                    o = o.view(num_views_per_clip, B, N, D).mean(dim=0)  # [B, N, D]
                else:
                    o = o  # Already [B, N, D]
                
                # Optional: add positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    o = o.reshape(B, T, S, D)
                    _indices = clip_indices[i][:, :: self.tubelet_size]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)
                    pos_embed = torch.gather(
                        pos_embed, 1,
                        _indices.unsqueeze(-1).expand(-1, -1, D)
                    )
                    pos_embed = pos_embed.unsqueeze(2).expand(-1, -1, S, -1)
                    o = o + pos_embed
                    o = o.flatten(1, 2)
                
                all_outputs.append(o)
            
            return all_outputs  # List of 4 tensors, each [B, N, D]
    
        return multiviews_postprocess(outputs)
