# evals/video_classification_frozen/modelcustom/videomae_encoder.py

import logging
import sys
import os
import warnings
from typing import Any, List, Union
from collections import OrderedDict

import torch
import torch.nn as nn

# Suppress timm registry overwrite warnings
warnings.filterwarnings("ignore", message="Overwriting .* in registry")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _collect_leaf_tensors(x: Any) -> List[torch.Tensor]:
    """Flattens nested list/tuple structures into a list of leaf tensors."""
    leaves: List[torch.Tensor] = []

    def rec(z: Any):
        if torch.is_tensor(z):
            leaves.append(z)
        elif isinstance(z, (list, tuple)):
            for zz in z:
                rec(zz)
        else:
            raise TypeError(f"Unsupported clip container type: {type(z)}")

    rec(x)
    if len(leaves) == 0:
        raise ValueError("No tensors found in clip container.")
    return leaves


class VideoMAEWrapper(nn.Module):
    """
    Wraps VideoMAE to satisfy V-JEPA2 eval API:
      - embed_dim attribute
      - forward(clips, clip_indices) returns Iterable[Tensor[B, N_tokens, D]]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        # Infer embed_dim from model
        self.embed_dim = getattr(model, "embed_dim", None)
        if self.embed_dim is None:
            if hasattr(model, "fc_norm") and hasattr(model.fc_norm, "weight"):
                self.embed_dim = int(model.fc_norm.weight.shape[0])
            elif hasattr(model, "norm") and hasattr(model.norm, "weight"):
                self.embed_dim = int(model.norm.weight.shape[0])
            else:
                raise ValueError("Could not infer embed_dim from model")
        
        logger.info(f"VideoMAEWrapper initialized with embed_dim={self.embed_dim}")

    @torch.no_grad()
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from VideoMAE.
        x: [B, C, T, H, W]
        returns: [B, N_tokens, D] or [B, 1, D] if pooled
        """
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            # Fallback: use forward but this might return logits
            feats = self.model(x)

        if feats.dim() == 2:
            feats = feats.unsqueeze(1)  # [B, 1, D]
        if feats.dim() != 3:
            raise ValueError(f"Expected 2D or 3D features, got shape={tuple(feats.shape)}")
        return feats

    def forward(
        self,
        clips: Union[torch.Tensor, List[Any]],
        clip_indices=None,
    ) -> List[torch.Tensor]:
        """Process nested clip structure and return per-clip features."""
        if torch.is_tensor(clips):
            x_flat = clips
            if x_flat.dim() == 4:
                raise ValueError(f"Unexpected tensor rank for clips: {x_flat.shape}")
            feats = self._forward_features(x_flat)
            return [feats]

        leaves = _collect_leaf_tensors(clips)
        B = int(leaves[0].shape[0])
        for t in leaves:
            if int(t.shape[0]) != B:
                raise ValueError("All clip leaves must share the same batch dimension.")

        x_flat = torch.cat(leaves, dim=0)
        feats_flat = self._forward_features(x_flat)

        num_leaves = len(leaves)
        N_tokens = int(feats_flat.shape[1])
        D = int(feats_flat.shape[2])

        feats = feats_flat.view(num_leaves, B, N_tokens, D).permute(1, 0, 2, 3).contiguous()
        return [feats[:, i, :, :] for i in range(num_leaves)]


def _import_modeling_finetune():
    """Dynamically import modeling_finetune from the vendored VideoMAE directory."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    videomae_dir = os.path.join(this_dir, "VideoMAE")
    
    if not os.path.isdir(videomae_dir):
        raise ImportError(f"VideoMAE directory not found at {videomae_dir}")
    
    if videomae_dir not in sys.path:
        sys.path.insert(0, videomae_dir)
        logger.info(f"Added {videomae_dir} to sys.path")
    
    import modeling_finetune
    return modeling_finetune


def _convert_pretrain_to_finetune_state_dict(pretrain_state_dict, model_state_dict):
    """
    Convert a VideoMAE PRETRAIN checkpoint to work with a FINETUNE model.
    
    Pretrain checkpoints have structure:
      - encoder.patch_embed.*, encoder.blocks.*, encoder.norm.*, encoder.pos_embed
      - decoder.*, decoder_pos_embed, mask_token, etc.
    
    Finetune models expect:
      - patch_embed.*, blocks.*, fc_norm.*, pos_embed, head.*
    
    Reference: MCG-NJU/VideoMAE run_class_finetuning.py lines 324-385
    """
    new_state_dict = OrderedDict()
    
    # Keys to skip (decoder-related, not needed for feature extraction)
    skip_prefixes = (
        "decoder", "mask_token", "decoder_pos_embed", 
        "decoder_embed", "decoder_blocks", "decoder_norm",
        "decoder_pred", "enc_dec_proj"
    )
    
    loaded_keys = []
    skipped_keys = []
    
    for key, value in pretrain_state_dict.items():
        original_key = key
        
        # Remove common prefixes
        if key.startswith("module."):
            key = key[7:]
        
        # Handle encoder. prefix (pretrain checkpoint format)
        if key.startswith("encoder."):
            key = key[8:]
        
        # Handle backbone. prefix (some checkpoint formats)
        if key.startswith("backbone."):
            key = key[9:]
        
        # Skip decoder-related keys
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_keys.append(original_key)
            continue
        
        # Handle norm -> fc_norm remapping (some VideoMAE versions)
        # The finetune model uses fc_norm instead of norm for final normalization
        if key == "norm.weight" and "fc_norm.weight" in model_state_dict:
            key = "fc_norm.weight"
        elif key == "norm.bias" and "fc_norm.bias" in model_state_dict:
            key = "fc_norm.bias"
        
        new_state_dict[key] = value
        loaded_keys.append(f"{original_key} -> {key}")
    
    # Remove head weights if they don't match (we'll use AttentiveClassifier instead)
    for head_key in ["head.weight", "head.bias"]:
        if head_key in new_state_dict:
            if head_key in model_state_dict:
                if new_state_dict[head_key].shape != model_state_dict[head_key].shape:
                    logger.info(f"Removing mismatched {head_key}: "
                              f"ckpt={new_state_dict[head_key].shape} vs model={model_state_dict[head_key].shape}")
                    del new_state_dict[head_key]
            else:
                del new_state_dict[head_key]
    
    logger.info(f"Checkpoint conversion: {len(loaded_keys)} keys converted, {len(skipped_keys)} decoder keys skipped")
    
    return new_state_dict


def _interpolate_pos_embed(pos_embed_ckpt, pos_embed_model, num_frames, tubelet_size=2):
    """
    Interpolate position embedding if spatial dimensions don't match.
    Reference: MCG-NJU/VideoMAE run_class_finetuning.py lines 358-380
    """
    if pos_embed_ckpt.shape == pos_embed_model.shape:
        return pos_embed_ckpt
    
    logger.info(f"Interpolating pos_embed: {pos_embed_ckpt.shape} -> {pos_embed_model.shape}")
    
    embedding_size = pos_embed_ckpt.shape[-1]
    num_patches_model = pos_embed_model.shape[1]
    num_extra_tokens = 0  # VideoMAE typically doesn't use cls token
    
    # Original spatial size
    temporal_patches = num_frames // tubelet_size
    orig_size = int(((pos_embed_ckpt.shape[1] - num_extra_tokens) / temporal_patches) ** 0.5)
    new_size = int((num_patches_model / temporal_patches) ** 0.5)
    
    if orig_size == new_size:
        return pos_embed_ckpt
    
    logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
    
    pos_tokens = pos_embed_ckpt[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, temporal_patches, orig_size, orig_size, embedding_size)
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, temporal_patches, new_size, new_size, embedding_size)
    pos_tokens = pos_tokens.flatten(1, 3)
    
    return pos_tokens


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    V-JEPA2 eval entrypoint for VideoMAE.
    
    Properly handles loading pretrain checkpoints into finetune model architecture.
    """
    logger.info(f"="*60)
    logger.info(f"Initializing VideoMAE encoder")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Frames per clip: {frames_per_clip}")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"="*60)

    enc_cfg = model_kwargs.get("encoder", model_kwargs)
    model_name = enc_cfg.get("model_name", None)
    
    if model_name is None:
        raise ValueError("VideoMAE config must include pretrain_kwargs.encoder.model_name")

    # Import VideoMAE from vendored location
    modeling_finetune = _import_modeling_finetune()

    if not hasattr(modeling_finetune, model_name):
        available = [n for n in dir(modeling_finetune) if n.startswith("vit_")]
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    ctor = getattr(modeling_finetune, model_name)
    extra_kwargs = {k: v for k, v in enc_cfg.items() if k != "model_name"}
    
    # VideoMAE model construction
    # The finetune models use 'all_frames' parameter
    tubelet_size = extra_kwargs.pop("tubelet_size", 2)
    
    model = None
    construction_attempts = [
        # Attempt 1: Full VideoMAE finetune style
        lambda: ctor(
            img_size=resolution,
            all_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            num_classes=1000,  # Placeholder, we won't use the head
            **extra_kwargs
        ),
        # Attempt 2: With num_frames instead
        lambda: ctor(
            img_size=resolution,
            num_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            **extra_kwargs
        ),
        # Attempt 3: Minimal (timm-style)
        lambda: ctor(**extra_kwargs),
    ]
    
    for i, attempt in enumerate(construction_attempts, 1):
        try:
            model = attempt()
            logger.info(f"Model construction succeeded on attempt {i}")
            break
        except TypeError as e:
            logger.debug(f"Attempt {i} failed: {e}")
            continue
    
    if model is None:
        raise RuntimeError(f"Could not construct model {model_name}")
    
    # Load checkpoint
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    logger.info(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
        logger.info("Found 'model' key in checkpoint")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        logger.info("Found 'state_dict' key in checkpoint")
    else:
        state_dict = ckpt
        logger.info("Using checkpoint directly as state_dict")
    
    # Log checkpoint structure for debugging
    sample_keys = list(state_dict.keys())[:10]
    logger.info(f"Checkpoint has {len(state_dict)} keys. Sample: {sample_keys}")
    
    # Check if this is a pretrain checkpoint (has encoder. prefix or decoder keys)
    has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())
    has_decoder_keys = any(k.startswith("decoder") for k in state_dict.keys())
    
    if has_encoder_prefix or has_decoder_keys:
        logger.info("Detected PRETRAIN checkpoint format - converting to finetune format")
        state_dict = _convert_pretrain_to_finetune_state_dict(state_dict, model.state_dict())
    else:
        logger.info("Detected FINETUNE checkpoint format - using directly")
        # Still need to strip module. prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Handle position embedding interpolation if needed
    if "pos_embed" in state_dict and "pos_embed" in model.state_dict():
        model_pos = model.state_dict()["pos_embed"]
        ckpt_pos = state_dict["pos_embed"]
        if ckpt_pos.shape != model_pos.shape:
            state_dict["pos_embed"] = _interpolate_pos_embed(
                ckpt_pos, model_pos, frames_per_clip, tubelet_size
            )
    
    # Load weights
    msg = model.load_state_dict(state_dict, strict=False)
    
    # Analyze loading results
    truly_missing = [k for k in msg.missing_keys if not k.startswith("head.")]
    
    logger.info(f"Weight loading complete:")
    logger.info(f"  - Missing keys: {len(msg.missing_keys)} (head-related: {len(msg.missing_keys) - len(truly_missing)})")
    logger.info(f"  - Unexpected keys: {len(msg.unexpected_keys)}")
    
    if truly_missing:
        logger.warning(f"  - Truly missing (non-head): {truly_missing[:5]}{'...' if len(truly_missing) > 5 else ''}")
    
    if msg.unexpected_keys:
        logger.debug(f"  - Unexpected: {msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")
    
    # Freeze encoder
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"VideoMAE encoder ready: {num_params:.1f}M parameters (frozen)")
    logger.info(f"="*60)

    return VideoMAEWrapper(model)