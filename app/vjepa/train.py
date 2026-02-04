# train.py  # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
import io
import boto3
import numpy as np
import torch
# import torch.multiprocessing as mp
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass


import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.checkpoint_loader import robust_checkpoint_loader

import torch.distributed as dist


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

# --
_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def prune_local_checkpoints(folder, max_to_keep=4):
    try:
        all_checkpoints = [f for f in os.listdir(folder) if f.startswith('e') and f.endswith('.pt')]
        if len(all_checkpoints) > max_to_keep:
            all_checkpoints.sort(key=lambda x: int(x[1:-3]))
            checkpoints_to_delete = all_checkpoints[:-max_to_keep]
            logger.info(
                f"Pruning local checkpoints. Keeping {max_to_keep}, deleting {len(checkpoints_to_delete)}."
            )
            for ckpt_name in checkpoints_to_delete:
                full_path = os.path.join(folder, ckpt_name)
                try:
                    os.remove(full_path)
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {full_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to prune checkpoints in folder {folder}: {e}")


def main(args, resume_preempt=False):
    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    s3_checkpoint_uri = cfgs_meta.get("s3_checkpoint_uri", None)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    save_every_steps = cfgs_meta.get("save_every_steps", 0)
    checkpoints_to_keep = cfgs_meta.get("checkpoints_to_keep", 3)
    # Legacy fallback
    max_epoch_checkpoints = cfgs_meta.get("max_epoch_checkpoints", checkpoints_to_keep)
    max_step_checkpoints = cfgs_meta.get("max_step_checkpoints", 5)
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    num_mask_tokens = cfgs_model.get("num_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    force_load_pretrain = cfgs_opt.get("force_load_pretrain", False)
    anneal_ckpt_path = cfgs_opt.get("anneal_ckpt", None)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if rank == 0 and s3_checkpoint_uri:
        logger.info(f"Checkpoints will be uploaded to: {s3_checkpoint_uri}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    encoder, predictor = init_video_model(
        device=device,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,  # int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    ## REVERTED: Create the target_encoder directly on the GPU using deepcopy.
    logger.info("Creating target_encoder via deepcopy on the GPU...")
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
    else:
        logger.info("Skipping model compilation.")

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches

    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    def make_momentum_scheduler(start_step=0):
        total = int(ipe * num_epochs * ipe_scale)
        return (
            ema[0] + i * (ema[1] - ema[0]) / total
            for i in range(start_step, total + 1)
        )

    # (LATEST) Do NOT instantiate momentum_scheduler yet; we must first decide
    # whether we're resuming or force-loading. We'll build it later
    # with the correct start_step so EMA is aligned.
    start_epoch, start_itr = 0, 0
    completed_steps = 0  # (LATEST) Track how many global steps have already been completed

    if force_load_pretrain:
        if anneal_ckpt_path and os.path.exists(anneal_ckpt_path):
            logger.info(f"FORCE-LOADING pretrained model from {anneal_ckpt_path}")
            checkpoint = robust_checkpoint_loader(anneal_ckpt_path, map_location=torch.device("cpu"))
            epoch_from_ckpt = checkpoint.get("epoch", 0)

            if "encoder" in checkpoint:
                pretrained_dict = checkpoint["encoder"]
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
                msg = encoder.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"Loaded pretrained encoder from epoch {epoch_from_ckpt} with msg: {msg}")

            if "target_encoder" in checkpoint:
                pretrained_dict = checkpoint["target_encoder"]
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
                msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"Loaded pretrained target encoder from epoch {epoch_from_ckpt} with msg: {msg}")

            if "predictor" in checkpoint:
                pretrained_dict = checkpoint["predictor"]
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
                msg = predictor.load_state_dict(pretrained_dict)
                logger.info(f"loaded pretrained predictor from epoch {epoch_from_ckpt} with msg: {msg}")

            del checkpoint
            gc.collect()
            logger.info("Force-loading of weights complete. Starting fresh from epoch 0.")
            completed_steps = 0  # (LATEST) Fresh start => EMA schedule begins at step 0
        else:
            logger.error(f"Configured to force-load but checkpoint was not found at: {anneal_ckpt_path}")
            raise FileNotFoundError(f"Anneal checkpoint not found: {anneal_ckpt_path}")
    else:
        latest_path = os.path.join(folder, "latest.pt")
        load_path = None

        if load_model or os.path.exists(latest_path):
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
            logger.info(f"Loading checkpoint from {load_path}")

            if load_path and os.path.exists(load_path):
                logger.info(f"Resuming training from checkpoint: {load_path}")
                (
                    encoder,
                    predictor,
                    target_encoder,
                    optimizer,
                    scaler,
                    start_epoch,
                    start_itr,
                ) = load_checkpoint(
                    r_path=load_path,
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    opt=optimizer,
                    scaler=scaler,
                )
                logger.info(f"SUCCESS: Loaded checkpoint from {load_path}")
                completed_steps = start_epoch * ipe + start_itr  # (LATEST) compute once

                # (LATEST) Burn LR/WD/Mask schedules up to the resume step, but do NOT
                # advance EMA here. We'll rebuild EMA to start at completed_steps.
                for _ in range(completed_steps):
                    scheduler.step()
                    wd_scheduler.step()
                    # next(momentum_scheduler)  # (LATEST) removed: do NOT consume EMA steps here
                    mask_collator.step()

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)

    ## REVERTED: Wrap the target_encoder in DDP as it now lives on the GPU.
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # (LATEST) Build EMA schedule only now, starting exactly at completed_steps.
    # - completed_steps == 0 for fresh/force-load
    # - completed_steps == start_epoch*ipe + start_itr for resume
    momentum_scheduler = make_momentum_scheduler(start_step=completed_steps)

    def save_checkpoint(epoch, itr, local_path, s3_uri_base=None, is_periodic=False):
        if rank != 0:
            return

        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "itr": itr,
        }

        try:
            torch.save(save_dict, local_path)
        except Exception as e:
            logger.error(f"Encountered exception when saving local checkpoint: {e}")
            return

        if s3_uri_base:
            try:
                s3_client = boto3.client("s3")
                bucket, key_prefix = s3_uri_base.replace("s3://", "").split("/", 1)
                filename = os.path.basename(local_path)
                s3_key = os.path.join(key_prefix, filename)

                # Check file size
                file_size = os.path.getsize(local_path)
                logger.info(f"Checkpoint size: {file_size / (1024**3):.2f} GB")

                if file_size > 5 * 1024**3:  # 5GB threshold
                    logger.info(f"Using multipart upload for large checkpoint...")
                    # Use multipart upload for files > 5GB
                    s3_client.upload_file(local_path, bucket, s3_key)
                else:
                    # Use regular upload for smaller files
                    with open(local_path, 'rb') as f:
                        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=f.read())
                logger.info(f"Successfully uploaded checkpoint to s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload checkpoint to S3. Error: {e}")

        if is_periodic and max_epoch_checkpoints > 0:
            prune_local_checkpoints(os.path.dirname(local_path), max_to_keep=max_epoch_checkpoints)

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    try:
        for epoch in range(start_epoch, num_epochs):
            unsupervised_sampler.set_epoch(epoch)
            logger.info("Epoch %d" % (epoch + 1))
    
            loss_meter = AverageMeter()
            mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
            iter_time_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            data_elapsed_time_meter = AverageMeter()
    
            itr_start = start_itr if epoch == start_epoch else 0
    
            for itr in range(itr_start, ipe):
                itr_start_time = time.time()
                iter_retries = 0
                iter_successful = False
    
                while not iter_successful:
                    try:
                        sample = next(loader)
                        iter_successful = True
                    except StopIteration:
                        logger.info("Exhausted data loaders. Refreshing...")
                        unsupervised_sampler.set_epoch(epoch)
                        loader = iter(unsupervised_loader)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.warning(
                                f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                            )
                            iter_retries += 1
                            time.sleep(5)
                            # refresh iterator (cheap) and try again
                            loader = iter(unsupervised_loader)
                        else:
                            logger.warning("Exceeded max retries; rebuilding DataLoader to respawn workers.")
                            (unsupervised_loader, unsupervised_sampler) = init_data(
                                data=dataset_type,
                                root_path=dataset_paths,
                                batch_size=batch_size,
                                training=True,
                                dataset_fpcs=dataset_fpcs,
                                fps=fps,
                                transform=transform,
                                rank=rank,
                                world_size=world_size,
                                datasets_weights=datasets_weights,
                                persistent_workers=persistent_workers,
                                collator=mask_collator,
                                num_workers=num_workers,
                                pin_mem=pin_mem,
                                log_dir=None,
                            )
                            unsupervised_sampler.set_epoch(epoch)
                            loader = iter(unsupervised_loader)
                            iter_retries = 0
                            # continue the while-loop instead of raising
                            continue
    
                for _fpc_sample in sample:
                    bs, fpc = _fpc_sample[0][-1][0].size()
                    mask_meters[fpc].update(bs / batch_size)
    
                def load_clips():
                    all_clips, all_masks_enc, all_masks_pred = [], [], []
                    for fpc_sample in sample:
                        udata, masks_enc, masks_pred = fpc_sample
                        all_clips += [udata[0][0].to(device, non_blocking=True)]
                        all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                        all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                    return all_clips, all_masks_enc, all_masks_pred
    
                clips, masks_enc, masks_pred = load_clips()
                data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
    
                if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                    logger.info("Running garbage collection...")
                    gc.collect()
    
                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
    
                    def forward_target(c):
                        with torch.no_grad():
                            ## REVERTED: No data transfer needed, both models are on the GPU
                            h = target_encoder(c)
                            h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                            return h
    
                    def forward_context(c):
                        z = encoder(c, masks_enc)
                        z = predictor(z, masks_enc, masks_pred)
                        return z
    
                    def loss_fn(z, h):
                        h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
                        loss, n = 0, 0
                        for zi, hi in zip(z, h):
                            for zij, hij in zip(zi, hi):
                                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                                n += 1
                        loss /= n
                        return loss
    
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        h = forward_target(clips)
                        z = forward_context(clips)
                        loss = loss_fn(z, h)
    
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
    
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
    
                    optimizer.zero_grad()
    
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        params_k = []
                        params_q = []
                        ## REVERTED: No device transfer needed for EMA update
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            params_k.append(param_k)
                            params_q.append(param_q)
                        torch._foreach_mul_(params_k, m)
                        torch._foreach_add_(params_k, params_q, alpha=1 - m)
    
                    return (
                        float(loss),
                        _new_lr,
                        _new_wd,
                    )
    
                (loss, _new_lr, _new_wd,), gpu_etime_ms = gpu_timer(train_step)
                iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
    
                loss_meter.update(loss)
                iter_time_meter.update(iter_elapsed_time_ms)
                gpu_time_meter.update(gpu_etime_ms)
                data_elapsed_time_meter.update(data_elapsed_time_ms)
    
                def log_stats():
                    csv_logger.log(
                        epoch + 1,
                        itr,
                        loss,
                        iter_elapsed_time_ms,
                        gpu_etime_ms,
                        data_elapsed_time_ms,
                    )
                    if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                        logger.info(
                            "[%d, %5d] loss: %.3f "
                            "masks: %s "
                            "[wd: %.2e] [lr: %.2e] "
                            "[mem: %.2e] "
                            "[iter: %.1f ms] "
                            "[gpu: %.1f ms] "
                            "[data: %.1f ms]"
                            % (
                                epoch + 1,
                                itr,
                                loss_meter.avg,
                                "["
                                + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters])
                                + "]",
                                _new_wd,
                                _new_lr,
                                torch.cuda.max_memory_allocated() / 1024.0**2,
                                iter_time_meter.avg,
                                gpu_time_meter.avg,
                                data_elapsed_time_meter.avg,
                            )
                        )
    
                log_stats()
                assert not np.isnan(loss), "loss is nan"
    
                # -- Step-based checkpoint saving with cleanup
                # if save_every_steps > 0 and (itr + 1) % save_every_steps == 0:
                #     # Only rank 0 should do cleanup to avoid race conditions
                #     if rank == 0:
                #         # Cleanup old step checkpoints FIRST - before saving new one
                #         step_pattern = os.path.join(folder, "step_*.pt")
                #         step_checkpoints = glob.glob(step_pattern)
                #         if len(step_checkpoints) >= max_step_checkpoints:  # Note: >= instead of >
                #             step_checkpoints.sort(key=os.path.getmtime, reverse=True)
                #             for old_checkpoint in step_checkpoints[max_step_checkpoints-1:]:  # Keep one less to make room
                #                 try:
                #                     os.remove(old_checkpoint)
                #                     logger.info(f"Removed old step checkpoint: {os.path.basename(old_checkpoint)}")
                #                 except OSError as e:
                #                     logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
                #
                #         # NOW save the new checkpoint (this already has rank check inside save_checkpoint)
                #         step_checkpoint_file = f"step_e{epoch}_i{itr}.pt"
                #         step_checkpoint_path = os.path.join(folder, step_checkpoint_file)
                #         save_checkpoint(epoch, itr, step_checkpoint_path, s3_checkpoint_uri)
                #         logger.info(f"Saved step checkpoint at epoch {epoch}, iteration {itr}")
    
            logger.info("avg. loss %.3f" % loss_meter.avg)
            _barrier()  # everyone reach end-of-epoch together
    
            latest_path = os.path.join(folder, "latest.pt")
            if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
                save_checkpoint(epoch + 1, 0, latest_path, None, is_periodic=False)
    
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, 0, save_every_path, s3_checkpoint_uri, is_periodic=True)
    
            _barrier()  # keep others from entering next epoch while rank 0 uploads
    finally:
        try:
            _barrier()
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
