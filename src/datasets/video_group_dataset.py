# src/datasets/video_group_dataset.py

import io
import math
import os
import pathlib
import warnings
from logging import getLogger

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

logger = getLogger()
MISSING_TOKEN = "MISS"


def _worker_init_fn(_):
    try:
        import torch as _torch, cv2, os as _os
        _torch.set_num_threads(1)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass


def make_videogroupdataset(
    *,
    data_paths,                  # str | list[str]  (CSV path(s))
    batch_size,
    group_size,                  # maps to num_segments from config
    frames_per_clip,
    frame_step=None,
    duration=None,
    fps=None,
    num_clips_per_video=1,       # NEW in your pipeline: per-video temporal clips
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    img_size=336,                      # <<< NEW (pass resolution)
    training=False,                 # <<< NEW
    miss_augment_prob=0.0,          # <<< NEW
    min_present=1,                  # <<< NEW
    split_name="train"
):
    ds = VideoGroupDataset(
        data_paths=data_paths,
        group_size=group_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=duration,
        fps=fps,
        num_clips_per_video=num_clips_per_video,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        img_size=img_size,             # <<< NEW
        training=training,                # <<< pass through
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name=split_name
    )
    
    # Mark the split (used by MISS augmentation)
    # ds._is_training = bool(training)

    # Optional per-worker resource logging, as in your other datasets
    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        ds = MonitoredDataset(
            dataset=ds,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoGroupDataset created")

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True
        )

    dl_kwargs = dict(
        dataset=ds,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1

    if deterministic:
        data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    else:
        data_loader = NondeterministicDataLoader(**dl_kwargs)

    logger.info("VideoGroupDataset data loader created")
    return ds, data_loader, dist_sampler


class VideoGroupDataset(Dataset):
    """
    One row per **study/group**. CSV must have:
      - a 'label' column (int)
      - N video columns for the group (any names). We will auto-detect them as
        all non-'label' columns in left-to-right order.
    Each of the N videos yields `num_clips_per_video` temporal clips.
    Total segments returned per sample = group_size * num_clips_per_video.

    S3 is supported via boto3; files are read into memory (no full local mirror).
    """

    def __init__(
        self,
        data_paths,
        group_size,
        frames_per_clip,
        frame_step=None,
        duration=None,
        fps=None,
        num_clips_per_video=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        shared_transform=None,
        transform=None,
        img_size=336,
        training=False, 
        miss_augment_prob=0.0, 
        min_present=1,
        split_name="train"
    ):
        super().__init__()
    
        # --- load & normalize CSVs (supports headerless or headered formats) ---
        def _read_group_csv(path: str) -> pd.DataFrame:
            # Try headerless, whitespace-delimited first
            try:
                df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
                if df.shape[1] == 1:
                    # fallback to "::" or single-space if needed
                    try:
                        df = pd.read_csv(path, header=None, sep="::", engine="python")
                    except Exception:
                        df = pd.read_csv(path, header=None, sep=" ", engine="python")
                ncols = df.shape[1]
                if ncols < 2:
                    raise ValueError(f"CSV '{path}' must have at least 2 columns (>=1 view + label)")
                view_cols = [f"view_{i}" for i in range(ncols - 1)]
                df.columns = view_cols + ["label"]
                return df
            except Exception:
                # Fallback: assume the file already has a header (must include 'label')
                df = pd.read_csv(path)
                if "label" not in df.columns:
                    raise ValueError(f"CSV '{path}' must contain a 'label' column or be headerless.")
                return df
    
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        dfs = [_read_group_csv(p) for p in data_paths]
        self.df = pd.concat(dfs, ignore_index=True)
    
        # Auto-detect video columns: everything except 'label'
        self.view_cols = [c for c in self.df.columns if c != "label"]
        if len(self.view_cols) == 0:
            raise ValueError("CSV must have at least one video column besides 'label'")
    
        # Enforce fixed group size deterministically
        self.view_cols = self.view_cols[:group_size]
        self.group_size = group_size
    
        # Core temporal / sampling configuration
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.duration = duration
        self.fps = fps
        self.num_clips_per_video = num_clips_per_video
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.shared_transform = shared_transform
        self.transform = transform
        
        self.img_size = int(img_size)
        self.miss_augment_prob = float(miss_augment_prob)
        self.min_present = int(min(min_present, self.group_size))
        self._is_training = bool(training)
        self.split_name = str(split_name)
      
        logger.info(
            f"[{self.split_name}] MISS augmentation: p={self.miss_augment_prob} "
            f"min_present={self.min_present} (train={self._is_training})"
        )
    
        # One S3 client per worker (lazily created in _ensure_s3_client)
        self.s3_client = None
    
        # Temporal mode validation (match VideoDataset semantics)
        if sum(v is not None for v in (self.fps, self.duration, self.frame_step)) != 1:
            raise ValueError(
                f"Must specify exactly one of fps={self.fps}, duration={self.duration}, or frame_step={self.frame_step}."
            )
    
        logger.info(f"Loaded {len(self.df)} groups; using columns: {self.view_cols}")


    def __len__(self):
        return len(self.df)

    # ---------- S3 helper ----------
    def _ensure_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3",
                config=Config(max_pool_connections=32, retries={"max_attempts": 5, "mode": "standard"}),
            )

    def _make_dummy_clip(self, fpc: int, h: int = 336, w: int = 336):
        """
        Create a black video clip [fpc, H, W, 3] to stand in for missing/failed views.
        Transforms will resize/crop as usual.
        """
        return np.zeros((fpc, h, w, 3), dtype=np.uint8)


    # ---------- Dataset API ----------
    def __getitem__(self, index):
        # retry semantics similar to VideoDataset
        while True:
            row = self.df.iloc[index]
            try:
                loaded = self._get_item_row(row)
                if loaded:
                    return loaded
            except Exception as e:
                warnings.warn(f"Retrying idx={index} due to error: {e}")
            index = np.random.randint(len(self))

    def _get_item_row(self, row):
        label = int(row["label"])
    
        # ---- collect URIs and initial presence flags from CSV ----
        uris, present = [], []
        for c in self.view_cols:
            v = row[c]
            if isinstance(v, str):
                v = v.strip()
            is_missing = (v is None) or (v == "") or (isinstance(v, float) and math.isnan(v)) or (v == MISSING_TOKEN)
            if is_missing:
                uris.append(None)
                present.append(0)
            else:
                uris.append(v)
                present.append(1)
    
        # ---- stochastic view-level MISS augmentation (training only) ----
        # Flip some PRESENT views to missing with prob self.miss_augment_prob,
        # while enforcing at least self.min_present survivors.
        if self._is_training and self.miss_augment_prob > 0.0:
            rng = np.random.default_rng()
            pres_idx = [i for i, p in enumerate(present) if p]
            if len(pres_idx) > 0:
                drops = rng.random(len(pres_idx)) < float(self.miss_augment_prob)
    
                # survivors if we applied the drops
                survivors = [i for i, d in zip(pres_idx, drops) if not d]
                need_min = max(1, int(getattr(self, "min_present", 1)))
                restore = set()
                if len(survivors) < need_min:
                    need = need_min - len(survivors)
                    dropped = [i for i, d in zip(pres_idx, drops) if d]
                    if len(dropped) > 0:
                        restore_sel = rng.choice(dropped, size=min(need, len(dropped)), replace=False)
                        restore = set(int(x) for x in np.atleast_1d(restore_sel))
    
                # apply the (possibly corrected) drops
                for i, d in zip(pres_idx, drops):
                    if d and (i not in restore):
                        uris[i] = None
                        present[i] = 0
    
        # ---- load/construct clips per slot ----
        segs, clip_indices_out, slot_mask = [], [], []
        for uri, p in zip(uris, present):
            if not p:
                # Missing view → dummy black clips (shape compatible with transforms)
                T = self.frames_per_clip
                H = W = self.img_size
                dummy = np.zeros((T, H, W, 3), dtype=np.uint8)
                clips = [dummy for _ in range(self.num_clips_per_video)]
                idxs  = [np.arange(T, dtype=np.int64) for _ in range(self.num_clips_per_video)]
            else:
                # Contiguous multi-clip loader (K clips of length fpc, non-overlapping)
                clips, idxs = self._loadvideo_decord_multi(uri, self.frames_per_clip, self.num_clips_per_video)
                if clips is None or len(clips) == 0:
                    # Fallback to dummy if load failed
                    T = self.frames_per_clip
                    H = W = self.img_size
                    dummy = np.zeros((T, H, W, 3), dtype=np.uint8)
                    clips = [dummy for _ in range(self.num_clips_per_video)]
                    idxs  = [np.arange(T, dtype=np.int64) for _ in range(self.num_clips_per_video)]
    
            if self.transform is not None:
                clips = [self.transform(c) for c in clips]
    
            segs.extend(clips)
            clip_indices_out.extend(idxs)
            # Per-clip presence flag (replicate the view's presence for its K clips)
            slot_mask.extend([bool(p)] * len(clips))
    
        return segs, label, clip_indices_out, torch.tensor(slot_mask, dtype=torch.bool)




    def _open_vr(self, sample_uri: str):
        # Local path
        if not (isinstance(sample_uri, str) and sample_uri.startswith("s3://")):
            if not os.path.exists(sample_uri):
                warnings.warn(f"video path not found fname='{sample_uri}'")
                return None
            if self.filter_long_videos:
                try:
                    _fsize = os.path.getsize(sample_uri)
                    if _fsize > self.filter_long_videos:
                        warnings.warn(f"skipping long video of size _fsize={_fsize} (bytes)")
                        return None
                except Exception:
                    pass
            try:
                return VideoReader(sample_uri, num_threads=-1, ctx=cpu(0))
            except Exception as e:
                logger.warning(f"VideoReader local fail: {e}")
                return None
    
        # S3 path
        try:
            bucket, key = sample_uri.replace("s3://", "").split("/", 1)
            self._ensure_s3_client()
    
            try:
                head = self.s3_client.head_object(Bucket=bucket, Key=key)
            except self.s3_client.exceptions.NoSuchKey:
                warnings.warn(f"video path not found fname='{sample_uri}'")
                return None
            except self.s3_client.exceptions.ClientError as e:
                logger.warning(f"S3 access error for {sample_uri}: {e}")
                return None
    
            fsize = head.get("ContentLength", 0)
            if self.filter_long_videos and fsize > self.filter_long_videos:
                warnings.warn(f"skipping long video of size _fsize={fsize} (bytes)")
                return None
    
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            if not data:
                logger.warning(f"Empty S3 object: {sample_uri}")
                return None
    
            bio = io.BytesIO(data)
            return VideoReader(bio, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            logger.warning(f"Failed to open video: {sample_uri}\n{e}")
            return None


    def _loadvideo_decord_multi(self, sample_uri: str, fpc: int, k: int):
        """
        Return K contiguous clips, each exactly fpc frames sampled at stride `fstp`,
        i.e., raw windows [i*clip_len, (i+1)*clip_len) without randomness.
    
        If the video is short, indices are clipped to [0, V-1] and padded by
        repeating the last valid frame to keep length fpc.
        """
        vr = self._open_vr(sample_uri)
        if vr is None:
            return [], None
    
        # --- derive stride and window size ---
        fstp = self.frame_step
        if (self.duration is not None) or (self.fps is not None):
            try:
                video_fps = max(1, int(math.ceil(vr.get_avg_fps())))
            except Exception as e:
                logger.warning(f"fps read failed: {e}")
                video_fps = None
    
            if self.duration is not None:
                assert self.fps is None
                if video_fps is None:
                    return [], None
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                assert self.duration is None
                if video_fps is None:
                    return [], None
                fstp = max(1, int(video_fps // max(1, self.fps)))
    
        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)          # raw-frame span per clip
        V = len(vr)                         # total raw frames
    
        # --- build K contiguous windows: [i*clip_len, (i+1)*clip_len) ---
        per_clip_inds = []
        for i in range(k):
            start = i * clip_len
            end   = start + clip_len
    
            # ideal regular sampling at stride `fstp`
            inds = np.arange(start, end, fstp, dtype=np.int64)  # length <= fpc
    
            # clamp to video range and pad if short
            if V > 0:
                inds = np.clip(inds, 0, V - 1)
            else:
                # empty video fallback -> all zeros
                inds = np.zeros((fpc,), dtype=np.int64)
    
            if inds.shape[0] < fpc:
                # pad by repeating last valid index
                pad = np.full((fpc - inds.shape[0],), inds[-1] if inds.shape[0] > 0 else 0, dtype=np.int64)
                inds = np.concatenate([inds, pad], axis=0)
    
            # defensively truncate (in case of off-by-one)
            if inds.shape[0] > fpc:
                inds = inds[:fpc]
    
            per_clip_inds.append(inds)
    
        # --- single batched fetch and split ---
        all_inds = np.concatenate(per_clip_inds, axis=0)
        frames_all = vr.get_batch(all_inds).asnumpy()  # [sum_k fpc, H, W, 3]
    
        clips = []
        offset = 0
        for _ in range(k):
            clips.append(frames_all[offset:offset + fpc])
            offset += fpc
    
        return clips, per_clip_inds



    # ---------- Sampling (shared with single-video dataset) ----------
    def _sample_from_vr(self, vr, fpc):
        """
        Returns (buffer[T,H,W,3], indices[np.int64, shape=(fpc,)])
        Picks a random valid window only when there is room to slide it.
        """
        # Resolve effective frame step
        fstp = self.frame_step
        if (self.duration is not None) or (self.fps is not None):
            try:
                video_fps = max(1, int(math.ceil(vr.get_avg_fps())))
            except Exception as e:
                logger.warning(f"fps read failed: {e}")
                video_fps = None
    
            if self.duration is not None:
                assert self.fps is None
                if video_fps is None:
                    raise RuntimeError("duration mode requires readable FPS")
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                assert self.duration is None
                if video_fps is None:
                    raise RuntimeError("fps mode requires readable FPS")
                fstp = max(1, int(video_fps // max(1, self.fps)))
    
        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)
        V = len(vr)
    
        if V < clip_len:
            # Too short → spread indices and pad to fpc if needed
            base = max(1, V // max(1, fstp))
            inds = np.linspace(0, max(0, V), num=base)
            if base < fpc:
                inds = np.concatenate((inds, np.ones(fpc - base) * max(0, V - 1)))
            inds = np.clip(inds, 0, max(0, V - 1)).astype(np.int64)
        else:
            # Enough frames; randomize only when there is slack
            if self.random_clip_sampling and (V > clip_len):
                # randint upper bound is EXCLUSIVE → use V+1 to allow end==V
                end_indx = np.random.randint(clip_len, V + 1)
            else:
                end_indx = clip_len  # single valid placement, no randomness
            start_indx = end_indx - clip_len
            inds = np.linspace(start_indx, end_indx, num=fpc)
            # Ensure [start, end) with integer frame indices
            inds = np.clip(inds, start_indx, max(start_indx, end_indx - 1)).astype(np.int64)
    
        buffer = vr.get_batch(inds).asnumpy()  # [T,H,W,3], uint8
        return buffer, inds


    def _split_into_clips(self, buffer, base_indices, fpc, num_clips):
        """
        Split a loaded video buffer into `num_clips` temporal clips of length fpc (frames),
        mirroring the main sampler’s semantics, and avoiding invalid randint ranges.
        Returns (clips: List[np.ndarray[T,H,W,3]], idx_slices: List[np.int64[fpc]]).
        """
        T = int(buffer.shape[0])
        if num_clips <= 1 or T <= fpc:
            # Not enough frames or no split requested
            inds = np.arange(min(fpc, T), dtype=np.int64)
            if len(inds) < fpc:
                pad = np.ones(fpc - len(inds), dtype=np.int64) * max(0, len(inds) - 1)
                inds = np.concatenate((inds, pad))
            return [buffer[inds]], [inds]
    
        partition_len = T // num_clips
        clips, idx_slices = [], []
    
        for i in range(num_clips):
            if partition_len > fpc:
                # Random window inside this partition only if there is slack
                end_indx = fpc
                if self.random_clip_sampling and (partition_len > fpc):
                    # EXCLUSIVE upper bound → +1 to allow end==partition_len
                    end_indx = np.random.randint(fpc, partition_len + 1)
                start_indx = end_indx - fpc
                inds = np.linspace(start_indx, end_indx, num=fpc)
                inds = np.clip(inds, start_indx, max(start_indx, end_indx - 1)).astype(np.int64)
                inds = inds + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    # Evenly spread within the partition; pad if needed
                    # Use step-aware count if frame_step is defined
                    step = max(1, (self.frame_step or 1))
                    base = max(1, partition_len // step)
                    inds = np.linspace(0, partition_len, num=base)
                    if base < fpc:
                        inds = np.concatenate((inds, np.ones(fpc - base) * max(0, partition_len - 1)))
                    inds = np.clip(inds, 0, max(0, partition_len - 1)).astype(np.int64)
                    inds = inds + i * partition_len
                else:
                    # Allow overlap across the whole sequence; slide partitions
                    sample_len = max(1, min(fpc, T) - 1)
                    base = max(1, sample_len)
                    inds = np.linspace(0, sample_len, num=base)
                    if base < fpc:
                        inds = np.concatenate((inds, np.ones(fpc - base) * sample_len))
                    inds = np.clip(inds, 0, sample_len).astype(np.int64)
                    clip_step = 0
                    if T > fpc and num_clips > 1:
                        clip_step = (T - fpc) // (num_clips - 1)
                    inds = inds + i * clip_step
    
            clips.append(buffer[inds])
            idx_slices.append(inds)
    
        return clips, idx_slices
