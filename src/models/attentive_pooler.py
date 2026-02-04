# src/models/attentive_pooler.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn

from src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, key_padding_mask: torch.Tensor | None = None):
        """
        x: [B, N, D]
        key_padding_mask: [B, N] (True = ignore / pad)
        """
        # For self-attn blocks (N->N), SDPA needs something broadcastable to [B, h, N, N]
        self_attn_mask = None
        if key_padding_mask is not None:
            # mask keys for all queries; broadcast over heads and query positions
            # self_attn_mask = key_padding_mask[:, None, None, :]  # [B,1,1,N], bool
            self_attn_mask = (~key_padding_mask)[:, None, None, :]  # [B,1,1,N]

        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    # pass (x, mask=None, attn_mask=self_attn_mask) positionally
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, None, self_attn_mask, use_reentrant=False
                    )
                else:
                    x = blk(x, mask=None, attn_mask=self_attn_mask)

        q = self.query_tokens.repeat(x.shape[0], 1, 1)

        # Cross-attn is Q->N; your CrossAttention expects [B,N], so keep the 2D mask here
        q = self.cross_attention_block(q, x, attn_mask=key_padding_mask)
        return q


class AttentiveClassifier(nn.Module):
    def __init__(self,
                 embed_dim=768, num_heads=12, mlp_ratio=4.0, depth=1,
                 norm_layer=nn.LayerNorm, init_std=0.02, qkv_bias=True,
                 num_classes=1000, complete_block=True,
                 use_activation_checkpointing=False,
                 # NEW (all optional; keep old defaults)
                 use_slot_embeddings: bool = False,
                 num_views: int = 9,
                 clips_per_view: int = 2,
                 use_factorized: bool = True):
        super().__init__()

        self.pooler = AttentivePooler(
            num_queries=1, embed_dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, depth=depth, norm_layer=norm_layer,
            init_std=init_std, qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

        # ---- NEW (but gated) ----
        self.use_slot_embeddings = bool(use_slot_embeddings)
        self.num_views = int(num_views)
        self.clips_per_view = int(clips_per_view)
        self.num_slots = self.num_views * self.clips_per_view
        self.use_factorized = bool(use_factorized)

        if self.use_slot_embeddings:
            expected = num_views * clips_per_view
            
            if self.use_factorized:
                self.view_embed = nn.Embedding(self.num_views, embed_dim)
                self.clip_embed = nn.Embedding(self.clips_per_view, embed_dim)
            else:
                self.slot_embed = nn.Embedding(self.num_slots, embed_dim)

    def _build_slot_emb(self, B, N, device):
        S = self.num_slots
        if N % S != 0:
            raise ValueError(f"N={N} not divisible by num_slots={S}")
        tokens_per_slot = N // S
        slot_ids = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1)  # [B,S]

        if self.use_factorized:
            view_ids = slot_ids // self.clips_per_view
            clip_ids = slot_ids %  self.clips_per_view
            v_emb = self.view_embed(view_ids)  # [B,S,D]
            c_emb = self.clip_embed(clip_ids) # [B,S,D]
            slot_emb = v_emb + c_emb
        else:
            slot_emb = self.slot_embed(slot_ids)  # [B,S,D]

        return slot_emb.repeat_interleave(tokens_per_slot, dim=1)  # [B,N,D]

    def forward(self, x, key_padding_mask=None):
        # Backwards compatible: if not using embeddings, behave exactly like before
        if self.use_slot_embeddings:
            B, N, D = x.shape
            x = x + self._build_slot_emb(B, N, x.device)
        x = self.pooler(x, key_padding_mask=key_padding_mask).squeeze(1)
        x = self.linear(x)
        return x


class AttentiveRegressor(nn.Module):  
    def __init__(self,
                 embed_dim=768, num_heads=12, mlp_ratio=4.0, depth=1,
                 norm_layer=nn.LayerNorm, init_std=0.02, qkv_bias=True,
                 num_targets=1, complete_block=True,
                 use_activation_checkpointing=False,
                 # NEW (Matched to Classifier)
                 use_slot_embeddings: bool = False,
                 num_views: int = 9,
                 clips_per_view: int = 2,
                 use_factorized: bool = True):
        super().__init__()

        self.pooler = AttentivePooler(
            num_queries=1, embed_dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, depth=depth, norm_layer=norm_layer,
            init_std=init_std, qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.regressor = nn.Linear(embed_dim, num_targets, bias=True)
        
        # ---- NEW (Mirrored from Classifier) ----
        self.use_slot_embeddings = bool(use_slot_embeddings)
        self.num_views = int(num_views)
        self.clips_per_view = int(clips_per_view)
        self.num_slots = self.num_views * self.clips_per_view
        self.use_factorized = bool(use_factorized)

        if self.use_slot_embeddings:
            expected = num_views * clips_per_view
            
            if self.use_factorized:
                self.view_embed = nn.Embedding(self.num_views, embed_dim)
                self.clip_embed = nn.Embedding(self.clips_per_view, embed_dim)
            else:
                self.slot_embed = nn.Embedding(self.num_slots, embed_dim)

    def _build_slot_emb(self, B, N, device):
        S = self.num_slots
        if N % S != 0:
            raise ValueError(
                f"Token count N={N} not divisible by num_slots S={S}. "
                f"Need early-fused tokens with stable slot ordering."
            )
    
        tokens_per_slot = N // S
        slot_ids = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1)  # [B,S]
    
        if self.use_factorized:
            view_ids = slot_ids // self.clips_per_view
            clip_ids = slot_ids %  self.clips_per_view
            slot_emb = self.view_embed(view_ids) + self.clip_embed(clip_ids)
        else:
            slot_emb = self.slot_embed(slot_ids)
    
        return slot_emb.repeat_interleave(tokens_per_slot, dim=1)  # [B,N,D]

      
    def forward(self, x, key_padding_mask=None):
        if self.use_slot_embeddings:
            B, N, D = x.shape
            x = x + self._build_slot_emb(B, N, x.device)
            
        x = self.pooler(x, key_padding_mask=key_padding_mask).squeeze(1)
        x = self.regressor(x)
        return x