# evals/video_classification_frozen_multi/eval.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
# except Exception:
#     pass

import logging
import math
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from evals.video_classification_frozen_multi.models import init_module
from evals.video_classification_frozen_multi.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier, AttentiveRegressor


from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

import os
import tempfile  # <-- ADD THIS

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/vjepa_run"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits; targets: [B] class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE (+ OPTIONAL ENV OVERRIDES)
    # ----------------------------------------------------------------------- #

    import os
    import inspect

    def set_override(env_var, target_dict, key, type_func=str):
        val = os.environ.get(env_var)
        if val is not None:
            if type_func == bool:
                val = val.lower() in ("true", "1", "t", "yes", "y")
            else:
                val = type_func(val)
            print(f"!!! MANUAL OVERRIDE: {key} -> {val}")
            target_dict[key] = val

    # Ensure nested dictionaries exist
    exp = args_eval.setdefault("experiment", {})
    clf = exp.setdefault("classifier", {})
    data = exp.setdefault("data", {})
    opt = exp.setdefault("optimization", {})

    # 1) Top-level overrides
    set_override("OVERRIDE_TAG", args_eval, "tag")
    set_override("OVERRIDE_VAL_ONLY", args_eval, "val_only", bool)
    set_override("OVERRIDE_PRED_PATH", args_eval, "predictions_save_path")
    set_override("OVERRIDE_CKPT", args_eval, "probe_checkpoint")

    # 2) Classifier overrides
    set_override("OVERRIDE_NUM_HEADS", clf, "num_heads", int)
    set_override("OVERRIDE_NUM_BLOCKS", clf, "num_probe_blocks", int)
    set_override("OVERRIDE_TASK_TYPE", clf, "task_type")              # "classification"|"regression"
    set_override("OVERRIDE_NUM_TARGETS", clf, "num_targets", int)     # regression only
    set_override("OVERRIDE_USE_SLOT_EMB", clf, "use_slot_embeddings", bool)
    set_override("OVERRIDE_USE_FACTORIZED", clf, "use_factorized", bool)

    # 3) Data overrides
    set_override("OVERRIDE_TRAIN_DATA", data, "dataset_train")
    set_override("OVERRIDE_VAL_DATA", data, "dataset_val")
    set_override("OVERRIDE_NUM_CLASSES", data, "num_classes", int)    # classification only
    set_override("OVERRIDE_RES", data, "resolution", int)
    set_override("OVERRIDE_NUM_SEGMENTS", data, "num_segments", int)
    set_override("OVERRIDE_FRAMES_PER_CLIP", data, "frames_per_clip", int)
    set_override("OVERRIDE_FRAME_STEP", data, "frame_step", int)
    set_override("OVERRIDE_NUM_CLIPS_PER_VIDEO", data, "num_clips_per_video", int)
    set_override("OVERRIDE_MISS_AUG_PROB", data, "miss_augment_prob", float)
    set_override("OVERRIDE_MIN_PRESENT", data, "min_present", int)
    
    # --- NEW: Override Mean/Std for regression ---
    set_override("OVERRIDE_TARGET_MEAN", data, "target_mean", float)
    set_override("OVERRIDE_TARGET_STD", data, "target_std", float)

    # 4) Optimization overrides
    set_override("OVERRIDE_EPOCHS", opt, "num_epochs", int)
    set_override("OVERRIDE_FOCAL_LOSS", opt, "use_focal_loss", bool)
    set_override("OVERRIDE_BATCH", opt, "batch_size", int)

    # In main(), after other set_override calls:
    set_override("OVERRIDE_LATE_FUSION", exp, "use_late_fusion", bool)

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")
    predictions_save_path = args_eval.get("predictions_save_path", None)

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    use_slot_embeddings = args_classifier.get("use_slot_embeddings", False)
    use_factorized = args_classifier.get("use_factorized", True)

    # Late fusion switch (default early fusion)
    use_late_fusion = args_exp.get("use_late_fusion", False) or args_classifier.get("use_late_fusion", False)

    # -- REGRESSION
    task_type = args_classifier.get("task_type", "classification")     # "classification" or "regression"
    num_targets = args_classifier.get("num_targets", None)             # regression only

    probe_checkpoint = args_eval.get("probe_checkpoint", None)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")

    num_classes = args_data.get("num_classes")                         # classification only
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]

    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # multi-specific
    num_clips_per_video = args_data.get("num_clips_per_video", 1)
    miss_augment_prob = args_data.get("miss_augment_prob", 0.0)
    min_present = args_data.get("min_present", 1)

    num_views = args_classifier.get("num_views", args_data.get("num_segments", 1))
    clips_per_view = args_classifier.get("clips_per_view", args_data.get("num_clips_per_video", 1))

    
    # --- NEW: Get Mean/Std from config ---
    target_mean = args_data.get("target_mean", None)
    target_std = args_data.get("target_std", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    use_focal_loss = args_opt.get("use_focal_loss", False)

    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")

    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")

    # Use custom probe checkpoint if specified, otherwise use default latest
    if probe_checkpoint is not None:
        latest_path = probe_checkpoint
    else:
        latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger
    if rank == 0:
        if task_type == "regression":
            csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_mae"), ("%.5f", "val_mae"))
        else:
            csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_acc"), ("%.5f", "val_acc"))

    # -- init encoder
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- init probe heads (classifier or regressor)
    def _filter_kwargs(cls, kwargs):
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}

    common_probe_kwargs = dict(
        embed_dim=encoder.embed_dim,
        num_heads=num_heads,
        depth=num_probe_blocks,
        use_activation_checkpointing=True,
        use_slot_embeddings=use_slot_embeddings,
        num_views=num_views,
        clips_per_view=clips_per_view,
        use_factorized=use_factorized,
    )

    if task_type == "regression":
        if num_targets is None:
            raise ValueError("task_type='regression' requires args_classifier['num_targets']")
        reg_kwargs = dict(common_probe_kwargs)
        reg_kwargs["num_targets"] = num_targets
        reg_kwargs = _filter_kwargs(AttentiveRegressor, reg_kwargs)
        classifiers = [AttentiveRegressor(**reg_kwargs).to(device) for _ in opt_kwargs]
    else:
        if num_classes is None:
            raise ValueError("task_type='classification' requires args_data['num_classes']")
        cls_kwargs = dict(common_probe_kwargs)
        cls_kwargs["num_classes"] = num_classes
        cls_kwargs = _filter_kwargs(AttentiveClassifier, cls_kwargs)
        classifiers = [AttentiveClassifier(**cls_kwargs).to(device) for _ in opt_kwargs]

    # -- DDP guard
    from torch import distributed as dist
    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1
    if use_ddp:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    else:
        logger.info(f"DDP disabled (world_size={world_size}); running single-process.")

    print(classifiers[0])

    # -- dataloaders
    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        num_clips_per_video=num_clips_per_video,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name="train",
    )

    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        num_clips_per_video=num_clips_per_video,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        miss_augment_prob=0.0,
        min_present=min_present,
        split_name="val",
    )

    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    # ---- per-head running stats ----
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0

    def save_checkpoint(
        epoch,
        mean_val_acc,
        best_val_acc,
        val_heads,
        best_per_head,
        mean_per_head,
        min_per_head,
        best_epoch_per_head,
        is_best=False,
    ):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if (scaler is None) else [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "mean_val_acc": float(mean_val_acc),
            "best_val_acc": float(best_val_acc),
            "val_acc_per_head": np.asarray(val_heads, dtype=float).tolist(),
            "best_val_acc_per_head": np.asarray(best_per_head, dtype=float).tolist(),
            "mean_val_acc_per_head": np.asarray(mean_per_head, dtype=float).tolist(),
            "min_val_acc_per_head": np.asarray(min_per_head, dtype=float).tolist(),
            "best_epoch_per_head": np.asarray(best_epoch_per_head, dtype=int).tolist(),
            "opt_grid": opt_kwargs,
            "task_type": task_type,
        }

        if rank == 0:
            _latest_path = os.path.join(folder, "latest.pt")
            torch.save(save_dict, _latest_path)

            epoch_path = os.path.join(folder, f"epoch_{epoch:03d}.pt")
            torch.save(save_dict, epoch_path)

            if is_best:
                best_path = os.path.join(folder, "best.pt")
                torch.save(save_dict, best_path)
                logger.info(f"Generated new best model: {best_path}")

    # [FIX] Initialize best scalar based on task
    if task_type == "regression":
        best_val_acc_scalar = float("inf")   # lower is better
    else:
        best_val_acc_scalar = 0.0            # higher is better

    val_cnt = 0
    val_sum_scalar = 0.0

    # --- ADD THIS SANITY CHECK ---
    if rank == 0:
        logger.info(f"Task Type: {task_type}")
        if task_type == "regression":
            logger.info(f"Regression Un-normalization: Mean={target_mean}, Std={target_std}")
    # -----------------------------

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if val_only:
            train_scalar, _ = -1.0, None
        else:
            train_scalar, _ = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                task_type=task_type,
                use_focal_loss=use_focal_loss,
                val_only=False,
                predictions_save_path=None,
                # --- NEW: Pass mean/std ---
                target_mean=target_mean,
                target_std=target_std,
                num_views=num_views,
                clips_per_view=clips_per_view,
                rank=rank,
                use_late_fusion=use_late_fusion,
            )

        val_scalar, val_heads = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            task_type=task_type,
            use_focal_loss=use_focal_loss,
            val_only=val_only,
            predictions_save_path=predictions_save_path,
            # --- NEW: Pass mean/std ---
            target_mean=target_mean,
            target_std=target_std,
            num_views=num_views,
            clips_per_view=clips_per_view,
            rank=rank,
            use_late_fusion=use_late_fusion,
        )

        # ---- update scalar running stats ----
        val_cnt += 1
        val_sum_scalar += float(val_scalar)
        mean_val_acc_scalar = val_sum_scalar / val_cnt

        # ---- determine "best" ----
        is_best = False
        if task_type == "regression":
            if float(val_scalar) < best_val_acc_scalar:
                best_val_acc_scalar = float(val_scalar)
                is_best = True
        else:
            if float(val_scalar) > best_val_acc_scalar:
                best_val_acc_scalar = float(val_scalar)
                is_best = True

        # ---- update per-head running stats ----
        count_epochs += 1
        if best_per_head is None:
            best_per_head = val_heads.copy()
            sum_per_head = val_heads.copy()
            min_per_head = val_heads.copy()
            best_epoch_per_head = np.full_like(val_heads, epoch + 1, dtype=int)
        else:
            if task_type == "regression":
                improved = val_heads < best_per_head
                best_per_head = np.minimum(best_per_head, val_heads)
            else:
                improved = val_heads > best_per_head
                best_per_head = np.maximum(best_per_head, val_heads)

            best_epoch_per_head[improved] = epoch + 1
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)

        mean_per_head = sum_per_head / count_epochs

        # ---- logging ----
        symbol = "" if task_type == "regression" else "%"
        val_label = "val(min-head)" if task_type == "regression" else "val(max-head)"

        logger.info(
            "[%5d] train: %.3f%s  %s: %.3f%s (Best: %.3f%s)"
            % (epoch + 1, train_scalar, symbol, val_label, val_scalar, symbol, best_val_acc_scalar, symbol)
        )

        if rank == 0:
            csv_logger.log(epoch + 1, train_scalar, val_scalar)

        if val_only:
            return

        save_checkpoint(
            epoch + 1,
            mean_val_acc_scalar,
            best_val_acc_scalar,
            val_heads,
            best_per_head,
            mean_per_head,
            min_per_head,
            best_epoch_per_head,
            is_best=is_best,
        )


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    task_type="classification",  # "classification" or "regression"
    use_focal_loss=False,
    val_only=False,
    predictions_save_path=None,
    # --- Arguments for un-normalization ---
    target_mean=None,
    target_std=None,
    num_views=None,
    clips_per_view=None,
    rank=None,
    # --- NEW: Late fusion ablation toggle ---
    use_late_fusion=False,
):
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    
    import inspect
    import numpy as np
    import os

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # Detect whether forward supports key_padding_mask (works for DDP too)
    supports_kpm = []
    for c in classifiers:
        mod = c.module if hasattr(c, "module") else c
        try:
            sig = inspect.signature(mod.forward)
            supports_kpm.append("key_padding_mask" in sig.parameters)
        except Exception:
            supports_kpm.append(False)

    for c in classifiers:
        c.train(mode=training)

    # Loss + meters
    if task_type == "regression":
        criterion = torch.nn.SmoothL1Loss()  # Huber
        metric_meters = [AverageMeter() for _ in classifiers]  # MAE meters
    else:
        criterion = FocalLoss(alpha=1.0, gamma=2.0) if use_focal_loss else torch.nn.CrossEntropyLoss()
        top1_meters = [AverageMeter() for _ in classifiers]

    all_predictions, all_video_paths, all_labels = [], [], []

    iterator = data_loader
    if val_only and tqdm is not None:
        iterator = tqdm(data_loader, desc="Inference", unit="batch", dynamic_ncols=True)

    # autocast: bfloat16 on CUDA only
    try:
        from torch.amp import autocast
        _has_torch_amp = True
    except Exception:
        from torch.cuda.amp import autocast
        _has_torch_amp = False

    from contextlib import nullcontext

    # Log fusion mode once at start
    if rank == 0:
        fusion_mode = "LATE FUSION (post-hoc averaging)" if use_late_fusion else "EARLY FUSION (token concatenation)"
        logger.info(f"Fusion mode: {fusion_mode}")

    for itr, data in enumerate(iterator):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        if device.type == "cuda":
            if _has_torch_amp:
                cast_ctx = autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16)
            else:
                cast_ctx = autocast(dtype=torch.bfloat16, enabled=use_bfloat16)
        else:
            cast_ctx = nullcontext()

        with cast_ctx:
            # ---- load batch ----
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            
            # --- DEBUG VERIFICATION ---
            if itr == 0:
                logger.info(f"VERIFICATION: Batch {itr}")
                logger.info(f"  > Loaded {len(clips)} distinct views/videos per patient.")
                logger.info(f"  > View 0 contains {len(clips[0])} temporal clips.")
                if len(clips[0]) > 0:
                    logger.info(f"  > Tensor Shape: {clips[0][0].shape}")
                
                total_clips = sum(len(v) for v in clips)
                logger.info(f"  > Total clips across all views: {total_clips}")
                logger.info(f"  > Expected by classifier: {num_views} × {clips_per_view} = {num_views * clips_per_view}")

            labels = data[1].to(device)
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]

            slot_present = None
            video_paths = None

            # your multi dataloader: data[3] is usually slot_present
            if len(data) > 3:
                if torch.is_tensor(data[3]):
                    slot_present = data[3].to(device)
                else:
                    video_paths = data[3]

            # if dataset provides paths separately, prefer that
            if len(data) > 4:
                video_paths = data[4]

            B = labels.shape[0]
            if video_paths is None:
                video_paths = [f"sample_{itr}_{i}" for i in range(B)]

            if itr == 0 and rank == 0:
                logger.info("video_paths (basenames): " + str([os.path.basename(p) for p in video_paths]))

            # ---- encoder forward (frozen) ----
            with torch.no_grad():
                enc_outs = encoder(clips, clip_indices)  # list of slot tensors, each [B, Nslot, D]

            if itr == 0:
                if (num_views is not None) and (clips_per_view is not None):
                    expected_slots = int(num_views) * int(clips_per_view)
                    if len(enc_outs) != expected_slots:
                        raise RuntimeError(
                            f"Encoder returned L={len(enc_outs)} slots, expected {expected_slots} "
                            f"(=num_views*clips_per_view)."
                        )

            # ---- probe forward ----
            with torch.set_grad_enabled(training):
                
                if use_late_fusion:
                    # ============================================================
                    # LATE FUSION: Process each slot independently, average predictions
                    # ============================================================
                    L = len(enc_outs)
                    Nslot = enc_outs[0].shape[1]
                    
                    # Build per-slot presence mask [B, L] for weighted averaging
                    if slot_present is not None:
                        slot_weights = slot_present.float()  # [B, V] or [B, L]
                        # Expand from per-view to per-slot if needed
                        if num_views is not None and clips_per_view is not None:
                            if slot_weights.shape[1] == num_views and L == num_views * clips_per_view:
                                slot_weights = slot_weights.repeat_interleave(clips_per_view, dim=1)  # [B, L]
                    else:
                        slot_weights = torch.ones(B, L, device=device)  # All present
                    
                    if itr == 0 and rank == 0:
                        logger.info(f"Late fusion: processing {L} slots independently")
                        logger.info(f"  > slot_weights shape: {tuple(slot_weights.shape)}")
                        logger.info(f"  > slot_weights[0]: {slot_weights[0].tolist()}")
                    
                    outs = []
                    for ci, c in enumerate(classifiers):
                        # Collect predictions from each slot
                        slot_preds = []
                        for slot_idx, slot_tokens in enumerate(enc_outs):
                            # slot_tokens: [B, Nslot, D]
                            pred = c(slot_tokens, key_padding_mask=None)  # [B, num_classes] or [B, num_targets]
                            slot_preds.append(pred)
                        
                        # Stack: [B, L, num_outputs]
                        stacked_preds = torch.stack(slot_preds, dim=1)
                        
                        # Weighted average (missing slots get weight 0)
                        # Normalize weights per sample
                        weights_sum = slot_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
                        normalized_weights = slot_weights / weights_sum  # [B, L]
                        
                        # Apply weights: [B, L, 1] * [B, L, num_outputs] -> sum over L
                        weighted_pred = (stacked_preds * normalized_weights.unsqueeze(-1)).sum(dim=1)  # [B, num_outputs]
                        outs.append(weighted_pred)
                    
                else:
                    # ============================================================
                    # EARLY FUSION: Concatenate all tokens, cross-attend jointly
                    # ============================================================
                    x = torch.cat(enc_outs, dim=1)  # [B, L*Nslot, D]
                    
                    # Build token-level padding mask aligned with the concatenation
                    key_padding_mask = None
                    if slot_present is not None:
                        if itr == 0 and rank == 0:
                            logger.info(f"slot_present shape: {tuple(slot_present.shape)}")
                            logger.info(f"L (enc_outs): {len(enc_outs)} | Nslot: {enc_outs[0].shape[1]}")

                        slot_keep = slot_present.bool()  # [B, V] or [B, L]
                        L = len(enc_outs)
                        Nslot = enc_outs[0].shape[1]

                        # If slot_present is per-view [B,V], expand to per-slot [B,L] where L=V*C
                        if num_views is not None and clips_per_view is not None:
                            if slot_keep.shape[1] == num_views and L == num_views * clips_per_view:
                                slot_keep = slot_keep.repeat_interleave(clips_per_view, dim=1)

                        if slot_keep.shape[1] != L:
                            raise RuntimeError(f"slot_present has {slot_keep.shape[1]} entries but encoder returned {L} slots")

                        token_keep = slot_keep.repeat_interleave(Nslot, dim=1)  # [B, L*Nslot]
                        key_padding_mask = ~token_keep  # True = ignore

                        if itr == 0 and rank == 0:
                            slot_masked = key_padding_mask[0].view(L, Nslot).all(dim=1)
                            logger.info(f"masked slots (per-slot): {slot_masked.tolist()}")

                    outs = []
                    for ci, c in enumerate(classifiers):
                        outs.append(c(x, key_padding_mask=key_padding_mask) if supports_kpm[ci] else c(x))

                # ---- Compute loss ----
                if task_type == "regression":
                    y = labels.float()
                    if y.dim() == 1:
                        y = y.unsqueeze(-1)
                    losses = [criterion(o.float(), y) for o in outs]
                else:
                    y = labels.long()
                    losses = [criterion(o, y) for o in outs]

        # ---- metrics + optional prediction collection ----
        with torch.no_grad():
            if task_type == "regression":
                mae_vals = [F.l1_loss(o.squeeze().float(), y.squeeze().float()) for o in outs]
                mae_vals = [float(AllReduce.apply(m)) for m in mae_vals]
                t_std = target_std if target_std is not None else 1.0
                mae_vals = [m * t_std for m in mae_vals]  # scale ONCE
                for meter, m in zip(metric_meters, mae_vals):
                    meter.update(m)

                if val_only and predictions_save_path is not None:
                    p0 = outs[0].detach().cpu().float().numpy()
                    y0 = y.detach().cpu().float().numpy()
                    for i in range(B):
                        all_predictions.append(p0[i])
                        all_video_paths.append(video_paths[i])
                        all_labels.append(y0[i])

                _agg = np.array([m.avg for m in metric_meters])

            else:
                top1 = [100.0 * o.argmax(dim=1).eq(y).float().mean() for o in outs]
                top1 = [float(AllReduce.apply(t)) for t in top1]
                for meter, a in zip(top1_meters, top1):
                    meter.update(a)

                if val_only and predictions_save_path is not None:
                    p0 = F.softmax(outs[0], dim=1).detach().cpu().numpy()
                    y0 = y.detach().cpu().numpy()
                    for i in range(B):
                        all_predictions.append(p0[i])
                        all_video_paths.append(video_paths[i])
                        all_labels.append(y0[i])

                _agg = np.array([m.avg for m in top1_meters])

        # ---- backward/step ----
        if training:
            for loss, opt in zip(losses, optimizer):
                loss.backward()
                opt.step()
                opt.zero_grad()

        # ---- periodic logging ----
        if itr % 10 == 0:
            if task_type == "regression":
                best_scalar = float(_agg.min())
                if val_only and hasattr(iterator, "set_description"):
                    iterator.set_description(f"Inf MAE: {best_scalar:.4f}")
                else:
                    logger.info(
                        "[%5d] %.4f [mean %.4f] [mem: %.2e]"
                        % (
                            itr,
                            best_scalar,
                            float(_agg.mean()),
                            (torch.cuda.max_memory_allocated() / 1024.0**2) if device.type == "cuda" else 0.0,
                        )
                    )
            else:
                best_scalar = float(_agg.max())
                if val_only and hasattr(iterator, "set_description"):
                    iterator.set_description(f"Inf Acc: {best_scalar:.2f}%")
                else:
                    logger.info(
                        "[%5d] %.3f%% [mean %.3f%% min %.3f%%] [mem: %.2e]"
                        % (
                            itr,
                            best_scalar,
                            float(_agg.mean()),
                            float(_agg.min()),
                            (torch.cuda.max_memory_allocated() / 1024.0**2) if device.type == "cuda" else 0.0,
                        )
                    )

    # ---- save predictions ----
    if val_only and predictions_save_path is not None and len(all_predictions) > 0:
        import pandas as pd

        out_dir = os.path.dirname(predictions_save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if task_type == "regression":
            t_mean = target_mean if target_mean is not None else 0.0
            t_std = target_std if target_std is not None else 1.0

            labels_real, preds_real = [], []

            for l in all_labels:
                l = np.asarray(l).reshape(-1)
                if l.size == 1:
                    labels_real.append(float(l[0] * t_std + t_mean))
                else:
                    labels_real.append((l * t_std + t_mean).tolist())

            for p in all_predictions:
                p = np.asarray(p).reshape(-1)
                if p.size == 1:
                    preds_real.append(float(p[0] * t_std + t_mean))
                else:
                    preds_real.append((p * t_std + t_mean).tolist())

            abs_err = None
            if len(labels_real) > 0 and isinstance(labels_real[0], (float, int)):
                abs_err = [abs(a - b) for a, b in zip(labels_real, preds_real)]

            df_dict = {
                "video_path": all_video_paths,
                "label_real": labels_real,
                "pred_real": preds_real,
            }
            if abs_err is not None:
                df_dict["abs_error"] = abs_err

            df = pd.DataFrame(df_dict)
        else:
            pred_classes = [int(np.argmax(p)) for p in all_predictions]
            pred_probs = [float(np.max(p)) for p in all_predictions]
            df = pd.DataFrame(
                {
                    "video_path": all_video_paths,
                    "true_label": [int(x) for x in all_labels],
                    "predicted_class": pred_classes,
                    "prediction_confidence": pred_probs,
                }
            )

        df.to_csv(predictions_save_path, index=False)
        logger.info(f"Saved {len(all_predictions)} predictions to {predictions_save_path}")

    scalar = float(_agg.min()) if task_type == "regression" else float(_agg.max())
    return scalar, _agg


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading classifier(s)
    pretrained_dict = checkpoint["classifiers"]
    
    # FIX: Handle DDP module prefix mismatch
    def fix_state_dict(state_dict, model):
        """Add or remove 'module.' prefix to match model structure."""
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Check if model expects module. prefix but checkpoint doesn't have it
        if any(k.startswith("module.") for k in model_keys) and not any(k.startswith("module.") for k in ckpt_keys):
            logger.info("Adding 'module.' prefix to checkpoint keys for DDP compatibility")
            return {"module." + k: v for k, v in state_dict.items()}
        
        # Check if checkpoint has module. prefix but model doesn't expect it
        if any(k.startswith("module.") for k in ckpt_keys) and not any(k.startswith("module.") for k in model_keys):
            logger.info("Removing 'module.' prefix from checkpoint keys")
            return {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        return state_dict
    
    pretrained_dict = [fix_state_dict(pd, c) for pd, c in zip(pretrained_dict, classifiers)]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
            logger.info(
                "loaded metrics: best_val_acc=%s mean_val_acc=%s",
                checkpoint.get("best_val_acc", "NA"),
                checkpoint.get("mean_val_acc", "NA"),
            )
        logger.info(f"loaded pretrained classifier (val_only) with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = int(checkpoint["epoch"])
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    # -- scaler (if used)
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        for s, sd in zip(scaler, checkpoint["scaler"]):
            if s is None or sd is None:
                continue
            s.load_state_dict(sd)

    if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
        logger.info(
            "loaded metrics: best_val_acc=%s mean_val_acc=%s",
            checkpoint.get("best_val_acc", "NA"),
            checkpoint.get("mean_val_acc", "NA"),
        )

    logger.info(f"loaded optimizers from epoch {epoch}")
    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    num_clips_per_video=1,  # NEW parameter  
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    miss_augment_prob=0.0,
    min_present=1,
    split_name="train"
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    # data_loader, dist_sampler = init_data(  
    #     data=dataset_type,  
    #     root_path=root_path,  
    #     batch_size=batch_size,  
    #     clip_len=frames_per_clip,  
    #     frame_sample_rate=frame_step,  
    #     duration=eval_duration,  
    #     num_clips=num_segments,  
    #     num_clips_per_video=num_clips_per_video,  # NEW: pass through  
    #     allow_clip_overlap=allow_segment_overlap,  
    #     transform=transform,  
    #     # shared_transform=shared_transform,  
    #     collator=collator,  
    #     num_workers=num_workers,  
    #     world_size=world_size,  
    #     rank=rank,  
    #     training=training,  
    #     pin_mem=True,  
    #     persistent_workers=True,  
    # )  

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        num_clips_per_video=num_clips_per_video,  # NEW: pass through  
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        img_size=img_size,
        training=training,
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name=split_name
    )
      
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]

        # Reference behavior: bf16 autocast does not require GradScaler
        scalers += [None]

    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd