import math

import timm
import torch
import torchvision

class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_parameter('pe', torch.nn.Parameter(torch.permute(pe, (1, 0, 2)), requires_grad=False))

    def forward(self, x):
        # Embed 1, 2, ..., n_frames -> higher dim w/ sin-cos embedding
        return self.dropout(x + self.pe[0, :x.size(1), :])

class ImageEncoder(torch.nn.Module):
    """2D image encoder."""
    def __init__(self, arch):
        super(ImageEncoder, self).__init__()

        if arch == 'resnet18':
            self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'resnet50':
            self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'convnext_tiny':
            self.model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            self.n_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Identity()
        elif arch == 'convnext_small':
            self.model = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
            self.n_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Identity()
        elif arch == 'convnext_base.fb_in22k_ft_in1k':
            self.model = timm.create_model(arch, pretrained=True)
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = torch.nn.Identity()
        elif arch == 'convnext_base.fb_in22k_ft_in1k_384':
            self.model = timm.create_model(arch, pretrained=True)
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = torch.nn.Identity()
        elif arch == 'swin_v2_s':
            self.model = torchvision.models.swin_v2_s(weights='IMAGENET1K_V1')
            self.n_features = self.model.head.in_features
            self.model.head = torch.nn.Identity()
        elif arch == 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k':
            self.model = timm.create_model(arch, pretrained=True)
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'echo-clip':
            from open_clip import create_model_and_transforms
            self.model, _, _ = create_model_and_transforms(
                f"hf-hub:mkaichristensen/{arch}", precision="bf16", device="cuda"
            )
            self.model = self.model.visual
            self.n_features = self.model.head.proj.out_features  # 512
        elif arch == 'convnextv2_tiny.fcmae_ft_in22k_in1k':
            self.model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = torch.nn.Identity()
        elif arch == 'convnextv2_tiny.fcmae':
            self.model = timm.create_model('convnextv2_tiny.fcmae', pretrained=True)
            self.n_features = 768
        else:
            import sys
            sys.exit('invalid --arch')

    def forward(self, x):
        return self.model(x)

class FrameTransformer(torch.nn.Module):
    """2+1D architecture with 2D image encoder and 1D temporal Transformer for echocardiogrpahy video modeling."""
    def __init__(self, arch, n_heads, n_layers, transformer_dropout, pooling, clip_len=16):
        super(FrameTransformer, self).__init__()
        self.pooling = pooling
        self.encoder = ImageEncoder(arch)

        transformer_encoder = torch.nn.TransformerEncoderLayer(d_model=self.encoder.n_features, nhead=n_heads, dim_feedforward=self.encoder.n_features, dropout=transformer_dropout, activation='relu', batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder, num_layers=n_layers)
 
        self.classifier = torch.nn.Identity()
    
        self.time_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=clip_len)

    def forward(self, x):
        b, c, l, h, w = x.shape

        # x: batch_size x 3 x clip_len x h x w
        x = x.reshape(b*l, c, h, w)  # batch_size*clip_len x 3 x h x w

        embeddings = self.encoder(x)

        # embeddings: batch_size x clip_len x n_features
        embeddings = embeddings.reshape(b, l, self.encoder.n_features)
        embeddings = self.time_encoder(embeddings)

        feats = self.transformer(embeddings)

        if self.pooling == 'mean':
            pooled_feats = feats.mean(1)
        elif self.pooling == 'mean-max':
            mean_pooled_feats = feats.mean(1)
            max_pooled_feats, _ = feats.max(1)
            pooled_feats = torch.cat([mean_pooled_feats, max_pooled_feats], dim=1)
        else:
            import sys
            sys.exit(-1)

        out = self.classifier(pooled_feats)

        return out

class MultiTaskModel(torch.nn.Module):
    """Multi-task model based on given encoder and task list."""
    def __init__(self, encoder, encoder_dim, tasks, fc_dropout=0, activations=True):
        super().__init__()
        self.encoder = encoder
        self.tasks = tasks
        self.activations = activations

        for task in self.tasks:
            if task.task_type == 'multi-class_classification':
                self.add_module(task.task_name+'_head', torch.nn.Sequential(torch.nn.Dropout(p=fc_dropout), torch.nn.Linear(encoder_dim, task.class_names.size)))
            else:  # binary classification or regression, both have 1 output
                self.add_module(task.task_name+'_head', torch.nn.Sequential(torch.nn.Dropout(p=fc_dropout), torch.nn.Linear(encoder_dim, 1)))
            
                # Initialize bias term to training set mean value (to account for different scales/units)
                self.get_submodule(task.task_name+'_head')[-1].bias.data[0] = task.mean
            
    def forward_features(self, x):
        x = self.encoder(x)

        return x

    def forward(self, x):
        x = self.encoder(x)

        out_dict = {}
        for task in self.tasks:
            out = self.get_submodule(task.task_name+'_head')(x)

            if self.activations:
                if task.task_type == 'binary_classification':
                    # Ensure that output corresponds to "positive" class for all binary classification tasks
                    if task.task_name in ['MVStenosis', 'AVStructure', 'RASize', 'RVSystolicFunction', 'LVWallThickness-increased-modsev', 'LVWallThickness-increased-any', 'pericardial-effusion']:
                        out_dict[task.task_name] = 1-torch.sigmoid(out)
                    else:
                        out_dict[task.task_name] = torch.sigmoid(out)
                elif task.task_type == 'multi-class_classification':
                    out_dict[task.task_name] = torch.softmax(out, dim=1)
                else:
                    out_dict[task.task_name] = out
            else:
                out_dict[task.task_name] = out

        return out_dict
