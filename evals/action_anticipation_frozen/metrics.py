# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F


class ClassMeanRecall:

    def __init__(self, num_classes: int, device: torch.device, k=5):
        self.num_classes = num_classes
        self.TP = torch.zeros(num_classes).to(device)
        self.FN = torch.zeros(num_classes).to(device)
        self.k = k

    def __call__(self, logits, labels, valid_classes=None, eps=1e-8):
        """
        :param logits: Tensors of shape [B, num_classes]
        :param labels: Tensors of shape [B]
        :param valid_classes: set
        """
        k, tp_tensor, fn_tensor = self.k, self.TP, self.FN
        logits = F.sigmoid(logits)

        if valid_classes is not None:
            _logits = torch.zeros(logits.shape).to(logits.device)
            for c in valid_classes:
                _logits[:, c] = logits[:, c]
            logits = _logits

        preds = logits.topk(k, dim=1).indices

        # Loop over batch and check whether all targets are within top-k logit
        # predictions for their respective class, if so TP else FN
        for p, gt in zip(preds, labels):
            if gt in p:
                tp_tensor[gt] += 1
            else:
                fn_tensor[gt] += 1

        # Aggregate TP/FN across all workers, but need to detach so that we
        # don't accidentally update tp_tensor and fn_tensor, which
        # only track local quantities.
        TP, FN = tp_tensor.clone(), fn_tensor.clone()
        dist.all_reduce(TP)
        dist.all_reduce(FN)

        nch = torch.sum((TP + FN) > 0)  # num classes hit; may not have TP/FP data for all classes yet
        recall = 100.0 * torch.sum(TP / (TP + FN + eps)) / nch  # mean class recall
        topk = 100.0 * sum(TP) / int(sum(TP + FN))  # accuracy

        return dict(
            recall=recall,
            accuracy=topk,
        )
