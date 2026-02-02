# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AttentiveClassifier(nn.Module):

    def __init__(
        self,
        verb_classes: dict,
        noun_classes: dict,
        action_classes: dict,
        embed_dim: int,
        num_heads: int,
        depth: int,
        use_activation_checkpointing: bool,
    ):
        super().__init__()
        self.num_verb_classes = len(verb_classes)
        num_noun_classes = len(noun_classes)
        num_action_classes = len(action_classes)
        self.action_only = self.num_verb_classes == 0

        self.pooler = AttentivePooler(
            num_queries=1 if self.action_only else 3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        if not self.action_only:
            self.verb_classifier = nn.Linear(embed_dim, self.num_verb_classes, bias=True)
            self.noun_classifier = nn.Linear(embed_dim, num_noun_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)

    def forward(self, x):
        if torch.isnan(x).any():
            print("Nan detected at output of encoder")
            exit(1)

        x = self.pooler(x)  # [B, 2, D]
        if not self.action_only:
            x_verb, x_noun, x_action = x[:, 0, :], x[:, 1, :], x[:, 2, :]
            x_verb = self.verb_classifier(x_verb)
            x_noun = self.noun_classifier(x_noun)
            x_action = self.action_classifier(x_action)
            return dict(
                verb=x_verb,
                noun=x_noun,
                action=x_action,
            )
        else:
            x_action = x[:, 0, :]
            x_action = self.action_classifier(x_action)
            return dict(action=x_action)


def init_module(
    module_name,
    device,
    frames_per_clip,
    frames_per_second,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for "model" module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])

      2) Needs to have a public attribute called 'embed_dim' (int) describing its
         output feature dimension.
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            frames_per_second=frames_per_second,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model


def init_classifier(
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    device: torch.device,
    num_classifiers: int,
    action_classes: dict,
    verb_classes: dict,
    noun_classes: dict,
):
    classifiers = [
        AttentiveClassifier(
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_blocks,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in range(num_classifiers)
    ]
    print(classifiers[0])
    return classifiers
