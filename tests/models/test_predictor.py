# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from src.models.predictor import VisionTransformerPredictor


class TestImagePredictorMaskTokens(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_dim = 768
        self._predictor = VisionTransformerPredictor(embed_dim=self._embed_dim, use_mask_tokens=True)

    def test_image_predictor_batchsize_4(self):
        BS = 4
        enc_mask_indices = [torch.tensor(BS * [[6, 7, 8]], dtype=torch.int64)]
        target_mask_indices = [torch.tensor(BS * [[16, 17, 18, 19]], dtype=torch.int64)]
        enc = torch.rand((BS, len(enc_mask_indices[0][0]), self._embed_dim))
        y = self._predictor(enc, enc_mask_indices, target_mask_indices)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, target_mask_indices[0].size(1), self._embed_dim))

    def test_image_predictor_batchsize_1(self):
        BS = 1
        enc_mask_indices = [torch.tensor(BS * [[6, 7, 8]], dtype=torch.int64)]
        target_mask_indices = [torch.tensor(BS * [[16, 17, 18, 19]], dtype=torch.int64)]
        enc = torch.rand((BS, len(enc_mask_indices[0][0]), self._embed_dim))
        y = self._predictor(enc, enc_mask_indices, target_mask_indices)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, target_mask_indices[0].size(1), self._embed_dim))


class TestVideoPredictorMaskTokens(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_dim = 768
        self._predictor = VisionTransformerPredictor(embed_dim=self._embed_dim, use_mask_tokens=True)

    def test_video_predictor_batchsize_4(self):
        BS = 4
        enc_mask_indices = [torch.tensor(BS * [[6, 7, 8]], dtype=torch.int64)]
        target_mask_indices = [torch.tensor(BS * [[16, 17, 18, 19]], dtype=torch.int64)]
        enc = torch.rand((BS, len(enc_mask_indices[0][0]), self._embed_dim))
        y = self._predictor(enc, enc_mask_indices, target_mask_indices)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, target_mask_indices[0].size(1), self._embed_dim))

    def test_video_predictor_batchsize_1(self):
        BS = 1
        enc_mask_indices = [torch.tensor(BS * [[6, 7, 8]], dtype=torch.int64)]
        target_mask_indices = [torch.tensor(BS * [[16, 17, 18, 19]], dtype=torch.int64)]
        enc = torch.rand((BS, len(enc_mask_indices[0][0]), self._embed_dim))
        y = self._predictor(enc, enc_mask_indices, target_mask_indices)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, target_mask_indices[0].size(1), self._embed_dim))
