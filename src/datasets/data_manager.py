# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True,
    subset_file=None,
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    num_clips_per_video=1,  # NEW parameter  
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,
    log_dir=None,
    img_size=336,
    miss_augment_prob=0.0,          # <<< NEW
    min_present=1,                  # <<< NEW
    split_name="train"
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
        )

    elif data.lower() == "videogroupdataset":  
        from src.datasets.video_group_dataset import make_videogroupdataset  
          
        dataset, data_loader, dist_sampler = make_videogroupdataset(  
            data_paths=root_path,  
            batch_size=batch_size,  
            group_size=num_clips,  # num_segments from config  
            frames_per_clip=clip_len,  
            frame_step=frame_sample_rate,  
            num_clips_per_video=num_clips_per_video,  # NEW  
            random_clip_sampling=random_clip_sampling,  
            allow_clip_overlap=allow_clip_overlap,  
            shared_transform=shared_transform,  
            transform=transform,  
            collator=collator,  
            num_workers=num_workers,  
            pin_mem=pin_mem,  
            persistent_workers=persistent_workers,  
            world_size=world_size,  
            rank=rank,  
            deterministic=deterministic,  
            log_dir=log_dir,
            img_size=img_size,                # <<< add this line
            training=training,                 # <<< NEW
            miss_augment_prob=miss_augment_prob,          # <<< NEW
            min_present=min_present,                  # <<< NEW
            split_name=split_name
        )

    return (data_loader, dist_sampler)
