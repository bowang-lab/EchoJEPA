# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

NAME = "vjepa2"
VERSION = "0.0.1"
DESCRIPTION = "PyTorch code and models for V-JEPA 2."
URL = "https://github.com/facebookresearch/vjepa2"


def get_requirements():
    with open("./requirements.txt") as reqsf:
        reqs = reqsf.readlines()
    return reqs


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        python_requires=">=3.11",
        install_requires=get_requirements(),
    )
