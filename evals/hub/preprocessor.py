# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def _make_transforms(crop_size=256):
    from ..video_classification_frozen.utils import make_transforms

    return make_transforms(crop_size=crop_size, training=False)


def vjepa2_preprocessor(*, pretrained: bool = True, **kwargs):
    crop_size = kwargs.get("crop_size", 256)
    return _make_transforms(crop_size=crop_size)
