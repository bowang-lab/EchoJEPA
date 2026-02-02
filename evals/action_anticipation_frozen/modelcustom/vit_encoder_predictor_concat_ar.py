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

import src.models.predictor as vit_pred
import src.models.vision_transformer as vit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    frames_per_clip: int,
    frames_per_second: int,
    resolution: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    # ----------------------------------------------------------------------- #
    # Initialize Encoder
    # ----------------------------------------------------------------------- #
    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    encoder = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    print(encoder)

    # ----------------------------------------------------------------------- #
    # Initialize Predictor
    # ----------------------------------------------------------------------- #
    prd_kwargs = model_kwargs["predictor"]
    prd_ckp_key = prd_kwargs.get("checkpoint_key")
    prd_model_name = prd_kwargs.get("model_name")

    predictor = vit_pred.__dict__[prd_model_name](
        img_size=resolution,
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **prd_kwargs,
    )
    pretrained_dict = checkpoint[prd_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in predictor.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict', pretrained_dict[k].shape, v.shape
            )
            pretrained_dict[k] = v
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor with msg: {msg}")
    print(predictor)

    # ----------------------------------------------------------------------- #
    # Build Wrapper
    # ----------------------------------------------------------------------- #
    model = AnticipativeWrapper(
        encoder=encoder,
        predictor=predictor,
        frames_per_second=frames_per_second,
        crop_size=resolution,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.embed_dim

    return model


class AnticipativeWrapper(torch.nn.Module):
    """Use predictor for inference"""

    def __init__(
        self,
        encoder,
        predictor,
        frames_per_second=4,
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
        # -- wrapper kwargs
        no_predictor=False,
        num_output_frames=2,
        num_steps=1,
        no_encoder=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = crop_size // patch_size
        self.tubelet_size = tubelet_size
        self.no_predictor = no_predictor
        self.num_output_frames = max(num_output_frames, tubelet_size)
        self.frames_per_second = frames_per_second
        self.num_steps = num_steps
        self.no_encoder = no_encoder

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"

    def forward(self, x, anticipation_times):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        :param anticipation_time: (Tensor) [B] seconds into the future to predict for each sample in batch
        """
        x = self.encoder(x)

        # determine 1D position of context tokens (x)
        # determine 1D position of prediction tokens
        # forward predictor with ctxt=x, tgt=None, masks_ctxt, masks_tgt
        if self.no_predictor:
            return x

        # Will output representations of $num_output_frames, that are
        # $anticipation_time seconds into the future.
        B, N, D = x.size()

        if self.no_encoder:
            x_accumulate = torch.rand(x.size(0), 0, x.size(2)).to(x.device)
        else:
            x_accumulate = x.clone()

        # Position IDs of the encoder patch tokens [B, N]
        ctxt_positions = torch.arange(N).unsqueeze(0).repeat(B, 1).to(x.device)

        # Position IDs of tokens to skip for each sample in batch [B]
        anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        skip_positions = N + int(self.grid_size**2) * anticipation_steps

        # Position IDs of tokens to predict [B, N_pred]
        N_pred = int(self.grid_size**2 * (self.num_output_frames // self.tubelet_size))
        tgt_positions = torch.arange(N_pred).unsqueeze(0).repeat(B, 1).to(x.device)
        tgt_positions += skip_positions.unsqueeze(1).repeat(1, N_pred)

        for _ in range(self.num_steps):
            x_pred = self.predictor(x, masks_x=ctxt_positions, masks_y=tgt_positions)
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x = torch.cat([x[:, N_pred:, :], x_pred], dim=1)

        return x_accumulate
