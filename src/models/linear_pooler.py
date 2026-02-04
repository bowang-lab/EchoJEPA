# src/models/linear_pooler.py
# Linear probe for standardized foundation model evaluation
#
# Drop-in replacement for AttentiveClassifier/AttentiveRegressor
# Uses mean pooling instead of learned cross-attention

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    Linear probe for classification tasks.
    
    Mean-pools input tokens and applies a linear classifier.
    Provides architecture-agnostic evaluation of frozen representations.
    
    Args:
        embed_dim: Dimension of input embeddings
        num_classes: Number of output classes
        use_layernorm: Whether to apply LayerNorm before classification (default: True)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1000,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        # Unused args for compatibility with AttentiveClassifier interface
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        norm_layer = nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        complete_block: bool = True,
        use_activation_checkpointing: bool = False,
        use_slot_embeddings: bool = False,
        num_views: int = 9,
        clips_per_view: int = 2,
        use_factorized: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Optional LayerNorm (helps with varying token distributions across models)
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Linear classifier
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, D]
            key_padding_mask: Optional mask of shape [B, N] where True = ignore/pad
            
        Returns:
            Logits of shape [B, num_classes]
        """
        B, N, D = x.shape
        
        if key_padding_mask is not None:
            # Masked mean pooling: only average over non-masked tokens
            # key_padding_mask: [B, N], True = ignore
            mask = ~key_padding_mask  # [B, N], True = keep
            mask = mask.unsqueeze(-1).float()  # [B, N, 1]
            
            # Sum of valid tokens
            x_sum = (x * mask).sum(dim=1)  # [B, D]
            
            # Count of valid tokens (avoid division by zero)
            counts = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            
            # Mean
            x_pooled = x_sum / counts  # [B, D]
        else:
            # Simple mean pooling
            x_pooled = x.mean(dim=1)  # [B, D]
        
        # Normalize, dropout, classify
        x_pooled = self.norm(x_pooled)
        x_pooled = self.dropout(x_pooled)
        logits = self.linear(x_pooled)
        
        return logits


class LinearRegressor(nn.Module):
    """
    Linear probe for regression tasks.
    
    Mean-pools input tokens and applies a linear regressor.
    Provides architecture-agnostic evaluation of frozen representations.
    
    Args:
        embed_dim: Dimension of input embeddings
        num_targets: Number of regression targets (default: 1)
        use_layernorm: Whether to apply LayerNorm before regression (default: True)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_targets: int = 1,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        # Unused args for compatibility with AttentiveRegressor interface
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        norm_layer = nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        complete_block: bool = True,
        use_activation_checkpointing: bool = False,
        use_slot_embeddings: bool = False,
        num_views: int = 9,
        clips_per_view: int = 2,
        use_factorized: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        
        # Optional LayerNorm
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Linear regressor
        self.regressor = nn.Linear(embed_dim, num_targets, bias=True)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.regressor.weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, D]
            key_padding_mask: Optional mask of shape [B, N] where True = ignore/pad
            
        Returns:
            Predictions of shape [B, num_targets]
        """
        B, N, D = x.shape
        
        if key_padding_mask is not None:
            # Masked mean pooling
            mask = ~key_padding_mask
            mask = mask.unsqueeze(-1).float()
            x_sum = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            x_pooled = x_sum / counts
        else:
            x_pooled = x.mean(dim=1)
        
        x_pooled = self.norm(x_pooled)
        x_pooled = self.dropout(x_pooled)
        output = self.regressor(x_pooled)
        
        return output


class MLPClassifier(nn.Module):
    """
    Two-layer MLP probe for classification.
    
    Slightly more expressive than linear while still being lightweight.
    Useful as a middle ground between linear and attentive probes.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1000,
        hidden_dim: int = None,
        use_layernorm: bool = True,
        dropout: float = 0.1,
        # Unused args for compatibility
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        **kwargs,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            mask = mask.unsqueeze(-1).float()
            x_sum = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            x_pooled = x_sum / counts
        else:
            x_pooled = x.mean(dim=1)
        
        x_pooled = self.norm(x_pooled)
        return self.mlp(x_pooled)


class MLPRegressor(nn.Module):
    """
    Two-layer MLP probe for regression.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_targets: int = 1,
        hidden_dim: int = None,
        use_layernorm: bool = True,
        dropout: float = 0.1,
        # Unused args for compatibility
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        **kwargs,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            mask = mask.unsqueeze(-1).float()
            x_sum = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            x_pooled = x_sum / counts
        else:
            x_pooled = x.mean(dim=1)
        
        x_pooled = self.norm(x_pooled)
        return self.mlp(x_pooled)