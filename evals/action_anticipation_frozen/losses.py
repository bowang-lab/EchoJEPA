# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="sum",
    detach=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    :param Tensor inputs: Prediction logits for each sample [B x K]
    :param Tensor targets: Class label for each sample [B] (long tensor)
    :param float alpha: Weight in range (0,1) to balance pos vs neg samples.
    :param float gamma: Exponent of modulating factor (1-p_t) to balance easy vs hard samples.
    :param str reduction: 'mean' | 'sum'
    """
    B, K = inputs.size()  # [batch_size, class logits]

    # convert to one-hot targets
    targets = F.one_hot(targets, K).float()  # [B, K]

    p = F.sigmoid(inputs)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    if detach:
        loss = loss.detach()

    return loss
