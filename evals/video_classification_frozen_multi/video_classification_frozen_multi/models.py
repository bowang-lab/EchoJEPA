# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    module_name,
    device,
    frames_per_clip,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for Encoder module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
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
