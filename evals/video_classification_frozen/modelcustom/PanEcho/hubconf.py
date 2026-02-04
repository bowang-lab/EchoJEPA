dependencies = ['torch', 'numpy', 'pandas']

import numpy as np
import pandas as pd
import torch

from src.models import FrameTransformer, MultiTaskModel

class Task():
    """Echocardiography interpretation task object."""
    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean

def PanEcho(pretrained=True, image_encoder_only=False, backbone_only=False, tasks='all', activations=True, clip_len=16):
    assert not (image_encoder_only and backbone_only), 'image_encoder_only and backbone_only cannot both be True'

    # PanEcho architecture specifications
    model_name = 'frame_transformer'
    arch = 'convnext_tiny'
    n_layers = 4
    n_heads = 8
    pooling = 'mean'
    transformer_dropout = 0.

    # Load tasks
    task_dict = pd.read_pickle('https://github.com/CarDS-Yale/PanEcho/blob/main/content/tasks.pkl?raw=true')
    all_tasks = list(task_dict.keys())
    task_list = [Task(t, task_dict[t]['task_type'], task_dict[t]['class_names'], task_dict[t]['mean']) for t in all_tasks]

    # Initialize model
    encoder = FrameTransformer(arch, n_heads, n_layers, transformer_dropout, pooling, clip_len)
    encoder_dim = encoder.encoder.n_features

    model = MultiTaskModel(encoder, encoder_dim, task_list, 0.25, activations)

    # Load pretrained weights
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt', map_location='cpu', progress=False)['weights']
        del weights['encoder.time_encoder.pe']  # allow for variable clip_len (fixed positional encoding does not need to be loaded from PanEcho)
        msg = model.load_state_dict(weights, strict=False)

    # Subset for desired tasks
    if tasks != 'all':
        for t in all_tasks:
            if t not in tasks:
                delattr(model, t+'_head')

        model.tasks = [t for t in task_list if t.task_name in tasks]

    # Return 2D image encoder only
    if image_encoder_only:
        model = model.encoder.encoder

    # Return 3D backbone (video encoder) only
    if backbone_only:
        model = model.encoder

    return model
