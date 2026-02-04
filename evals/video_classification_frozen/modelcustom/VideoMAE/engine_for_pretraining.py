import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import contextlib

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, update_freq: int = 1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    if update_freq < 1:
        update_freq = 1
    
    optimizer.zero_grad(set_to_none=True)
    
    num_micro_steps = len(data_loader)
    num_update_steps = num_micro_steps // update_freq
    max_micro_steps = num_update_steps * update_freq
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step >= max_micro_steps:
            break

        update_step = step // update_freq
        it = start_steps + update_step
    
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
    
        videos, bool_masked_pos, paths = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
    
        with torch.no_grad():
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean
    
            if normlize_target:
                videos_squeeze = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2, p1=patch_size, p2=patch_size
                )
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)) / (
                    videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                )
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2, p1=patch_size, p2=patch_size
                )
    
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
    
        # --- FIX: Updated deprecated autocast syntax ---
        with torch.amp.autocast('cuda'):
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)
    
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
    
        loss = loss / update_freq
        update_grad = (step + 1) % update_freq == 0
    
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    
        sync_ctx = contextlib.nullcontext()
        if hasattr(model, "no_sync") and not update_grad:
            sync_ctx = model.no_sync()
    
        with sync_ctx:
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(), create_graph=is_second_order,
                update_grad=update_grad
            )
        
        if update_grad:
            if grad_norm is None or (isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm)):
                print("[WARN] non-finite grad_norm; cleared grads and continuing", flush=True)
            optimizer.zero_grad(set_to_none=True)

        if update_grad and (grad_norm is not None) and (not torch.isfinite(grad_norm)):
            print("[BAD BATCH PATHS]", paths, flush=True)

    
        torch.cuda.synchronize()
    
        metric_logger.update(loss=loss_value)

        # --- FIX: Log LR every step (avoids ZeroDivisionError on skipped steps) ---
        min_lr = min(pg["lr"] for pg in optimizer.param_groups)
        max_lr = max(pg["lr"] for pg in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        
        if update_grad:
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
    
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
    
            if grad_norm is not None:
                metric_logger.update(grad_norm=grad_norm)
    
            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                if grad_norm is not None:
                    log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}