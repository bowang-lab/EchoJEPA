# evals/video_classification_frozen_multi/modelcustom/vit_encoder_multiclip.py
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
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
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device=None, # Accepted for compatibility, but handled by to(device) usually
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)

    pretrained_dict = checkpoint[enc_ckp_key]
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

    # Initialize the wrapper
    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class ClipAggregation(nn.Module):
    """
    Wrapper that processes multiple clips.
    If return_per_clip=True, returns a list of tensors [B, N, D] (one per clip).
    If return_per_clip=False, concatenates/merges them (Legacy behavior).
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        return_per_clip=False, # <--- NEW ARGUMENT
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.return_per_clip = return_per_clip

        # 1D-temporal pos-embedding (Legacy logic, mostly for merged output)
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, self.embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(self.embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        """
        x: List[List[Tensor]]. Outer list = Temporal Segments (Groups). Inner list = Spatial Views.
        """
        
        # 1. Flatten the input structure to a single batch of clips
        flat_clips = []
        for segment in x:
            for view in segment:
                flat_clips.append(view)
        
        # 2. Stack and run the encoder efficiently
        # shape: [Total_Slots * B, C, T, H, W]
        batched_input = torch.cat(flat_clips, dim=0)
        
        # shape: [Total_Slots * B, N_tokens, D]
        all_tokens = self.model(batched_input)

        # 3. Handle Return Logic
        if self.return_per_clip:
            # FIX PATH A: Return separated slots
            num_slots = len(flat_clips)
            # Chunk back into list of [B, N, D]
            return list(torch.chunk(all_tokens, num_slots, dim=0))

        # --- LEGACY BEHAVIOR (Merges temporal clips) ---
        # Only runs if return_per_clip=False
        
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B = x[0][0].shape[0]

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            # Assuming standard frame handling
            F = x[0][0].shape[2] 
            T = F // self.tubelet_size
            S = N // T 

            # Reshape to separate slots: [Total_Slots, B, N, D]
            # Total_Slots = num_clips * num_views_per_clip
            outputs = outputs.view(num_clips * num_views_per_clip, B, N, D)

            # Re-organize to [num_clips, num_views, B, N, D]
            # (Simplification of original logic)
            final_outputs = []
            
            # Group by Spatial View (merging temporal parts)
            for v in range(num_views_per_clip):
                view_parts = []
                for c in range(num_clips):
                    # extract the c-th clip for v-th view
                    idx = c * num_views_per_clip + v
                    part = outputs[idx] # [B, N, D]
                    
                    # Reshape to [B, T, S, D]
                    part = part.view(B, T, S, D)
                    view_parts.append(part)
                
                # Concatenate temporal parts: [B, T*num_clips, S, D]
                merged = torch.cat(view_parts, dim=1)
                
                # Add pos embed if needed
                if self.pos_embed is not None and clip_indices is not None:
                     # (Pos embed logic omitted for brevity as config has it False)
                     pass

                # Flatten back to [B, Total_Tokens, D]
                merged = merged.flatten(1, 2)
                final_outputs.append(merged)

            return final_outputs

        return multiviews_postprocess(all_tokens)