import os
import gc
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tqdm

from copy import deepcopy

from sklearn import metrics

from ddp_utils import all_gather, is_main_process

class Task():
    """Echocardiography interpretation task object."""
    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean

def merge_task_dicts(d):
    merged_dict = {}

    # Iterate through each dictionary in the list
    for dictionary in d:
        # Iterate through keys in the dictionary
        for key, value in dictionary.items():
            if key in merged_dict:
                # Merge lists if key is present in both merged_dict and the current dictionary
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in merged_dict[key]:
                            merged_dict[key][sub_key] += sub_value
                        else:
                            merged_dict[key][sub_key] = sub_value
                else:
                    merged_dict[key] += value
            else:
                # Otherwise, assign the corresponding dictionary
                merged_dict[key] = value

    return merged_dict

def time_elapsed(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f'{hours:.0f}h:{minutes:.0f}m:{seconds:.0f}s'

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, amp=False):
    data_loader.sampler.set_epoch(epoch)  # for DDP
    
    model.train()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'video_nums': [], 'views': [], 'invalid_batches': 0} for task in tasks}
    for b, batch in pbar:
        x = batch['x'].cuda(memory_format=torch.channels_last_3d)
        acc_num = batch['acc_num']
        video_num = batch['video_num']
        view = batch['view']

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            # Forward pass
            out_dict = model(x)

            # Compute loss for each task
            losses = []
            for task in tasks:
                yhat = out_dict[task.task_name]
                y = batch[task.task_name].cuda()
                mask = batch[task.task_name+'_mask'].cuda()

                # If batch contains *only* missing values for task, then skip loss computation
                if mask.sum() == 0:
                    task_data[task.task_name]['invalid_batches'] += 1
                    continue

                # Mask out missing values from loss computation
                masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                # Collect (masked) true and predicted labels
                if task.task_type == 'multi-class_classification':
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                elif task.task_type == 'binary_classification':
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                else:
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                # Collect (masked) auxiliary information
                masked_acc_num = [a for a, m in zip(acc_num, mask) if m]
                masked_video_num = [v for v, m in zip(video_num, mask) if m]
                masked_view = [v for v, m in zip(view, mask) if m]
                task_data[task.task_name]['acc_nums'].append(masked_acc_num)
                task_data[task.task_name]['video_nums'].append(masked_video_num)
                task_data[task.task_name]['views'].append(masked_view)

                # For CrossEntropyLoss, target must have shape (N,)
                if task.task_type == 'multi-class_classification':
                    masked_y = masked_y.squeeze(1)

                # Compute task loss
                loss = loss_fxns[task.task_name](masked_yhat, masked_y)
                # Scale down regression loss based on mean value
                if task.task_type == 'regression':
                    loss /= task.mean
                losses.append(loss)

                # Keep track of task losses for each batch
                task_data[task.task_name]['losses'].append(loss.item())
                del loss
            # Compute overall loss
            if len(losses) == 0:
                continue
            else:
                loss = sum(losses) / len(losses)
                del losses
                
                # Keep running sum of losses for each batch
                running_loss += loss.item()
            overall_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, mask, out_dict

    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.array(overall_losses).ravel()

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'train'
    current_metrics_dict['loss'] = overall_losses.mean()
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
            'video_num': np.concatenate(task_data[task.task_name]['video_nums'], axis=0),
            'view': np.concatenate(task_data[task.task_name]['views'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
        
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    current_metrics_dict['mean_auroc_r2'] = (current_metrics_dict['mean_classification_auroc'] + current_metrics_dict['mean_regression_r2']) / 2
    print(f'[OVERALL] Mean (AUROC, R^2): {current_metrics_dict["mean_auroc_r2"]:.3f}')
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    gc.collect()

    return pd.concat([history, current_metrics], axis=0)

def validate(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp=False, scheduler=None):
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'video_nums': [], 'views': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            acc_num = batch['acc_num']
            video_num = batch['video_num']
            view = batch['view']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()

                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_acc_num = [a for a, m in zip(acc_num, mask) if m]
                    masked_video_num = [v for v, m in zip(video_num, mask) if m]
                    masked_view = [v for v, m in zip(view, mask) if m]
                    task_data[task.task_name]['acc_nums'].append(masked_acc_num)
                    task_data[task.task_name]['video_nums'].append(masked_video_num)
                    task_data[task.task_name]['views'].append(masked_view)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](masked_yhat, masked_y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # rough estimate using 1/n_gpu of the data

    del x, mask, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.concatenate(all_gather(overall_losses)).ravel()
    val_loss = overall_losses.mean()
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'val'
    current_metrics_dict['loss'] = val_loss
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()  # mean reduce
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
            'video_num': np.concatenate(task_data[task.task_name]['video_nums'], axis=0),
            'view': np.concatenate(task_data[task.task_name]['views'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
                    
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    current_metrics_dict['mean_auroc_r2'] = (current_metrics_dict['mean_classification_auroc'] + current_metrics_dict['mean_regression_r2']) / 2
    print(f'[OVERALL] Mean (AUROC, R^2): {current_metrics_dict["mean_auroc_r2"]:.3f}')
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping by mean of AUROC and R2 (since both increasing metrics bounded in [0, 1])
    if current_metrics_dict['mean_auroc_r2'] > early_stopping_dict['best_mean_auroc_r2']:
        print(f'EARLY STOPPING: mean_auroc_r2 has improved from {early_stopping_dict["best_mean_auroc_r2"]:.3f} to {current_metrics_dict["mean_auroc_r2"]:.3f}! Saving weights.')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_mean_auroc_r2'] = current_metrics_dict['mean_auroc_r2']
        best_model_wts = deepcopy(model.module.state_dict()) if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel) else deepcopy(model.state_dict())
        if is_main_process():
            torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        early_stopping_dict['epochs_no_improve'] += 1
        print(f'EARLY STOPPING: mean_auroc_r2 has not improved from {early_stopping_dict["best_mean_auroc_r2"]:.3f} since epoch {epoch-early_stopping_dict["epochs_no_improve"]}.')

    dist.barrier()

    # Apply learning rate scheduler (if given)
    if scheduler is not None:
        scheduler.step(current_metrics_dict['mean_auroc_r2'])

    history = pd.concat([history, current_metrics], axis=0)
    # Plot current metrics
    if is_main_process():
        for col in np.setdiff1d(history.columns.values, ['epoch', 'phase']):
            task_name = col.split('_')[0]

            sub_history = history.dropna(subset=col).reset_index(drop=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(sub_history.loc[sub_history['phase'] == 'train', 'epoch'], sub_history.loc[sub_history['phase'] == 'train', col], marker='o', label='train')
            ax.plot(sub_history.loc[sub_history['phase'] == 'val', 'epoch'], sub_history.loc[sub_history['phase'] == 'val', col], marker='o', label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.legend()
            if task_name in [t.task_name for t in tasks]:
                fig.savefig(os.path.join(model_dir, 'history_plots', task_name, f'{col}_history.png'), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(model_dir, 'history_plots', f'{col}_history.png'), dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close(fig)

    gc.collect()

    return history, early_stopping_dict, best_model_wts

def evaluate(model, tasks, loss_fxns, data_loader, split, history, model_dir, weights, amp=False):
    # Load weights determined by early stopping
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        msg = model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights)
    print(msg)
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'video_nums': [], 'views': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            acc_num = batch['acc_num']
            video_num = batch['video_num']
            view = batch['view']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()

                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_acc_num = [a for a, m in zip(acc_num, mask) if m]
                    masked_video_num = [v for v, m in zip(video_num, mask) if m]
                    masked_view = [v for v, m in zip(view, mask) if m]
                    task_data[task.task_name]['acc_nums'].append(masked_acc_num)
                    task_data[task.task_name]['video_nums'].append(masked_video_num)
                    task_data[task.task_name]['views'].append(masked_view)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](masked_yhat, masked_y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, mask, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    out_str = ''
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            out_str += f'--- {task.task_name} [{task.task_type}] (N=0) ---\n'
            continue

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
            'video_num': np.concatenate(task_data[task.task_name]['video_nums'], axis=0),
            'view': np.concatenate(task_data[task.task_name]['views'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        out_str += f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---\n'

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    fpr, tpr, _ = metrics.roc_curve(binary_y, yhat[:, class_idx])
                    prs, res, _ = metrics.precision_recall_curve(binary_y, yhat[:, class_idx])

                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                    # Plot class-specific ROC curve
                    roc_ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUROC: {auroc:.3f})')
                    # Plot class-specific PR curve
                    p = pr_ax.plot(res, prs, lw=2, label=f'{class_name} (AP: {ap:.3f})')
                    pr_ax.axhline(y=binary_y.sum()/binary_y.size, color=p[0].get_color(), lw=2, linestyle='--')

                    aurocs.append(auroc)
                    aps.append(ap)
                out_str += f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            out_str += f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)

        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

                fpr, tpr, _ = metrics.roc_curve(y, yhat)
                prs, res, _ = metrics.precision_recall_curve(y, yhat)

                # Plot class-specific ROC curve
                roc_ax.plot(fpr, tpr, lw=2, label=f'{task.class_names[1]} (AUROC: {auroc:.3f})')
                # Plot class-specific PR curve
                p = pr_ax.plot(res, prs, lw=2, label=f'{task.class_names[1]} (AP: {ap:.3f})')
                pr_ax.axhline(y=y.sum()/y.size, color=p[0].get_color(), lw=2, linestyle='--')
            
            out_str += f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(auroc)
            classification_aps.append(ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)
            
            out_str += f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}\n'

            # Keep track of overall mean classification metrics
            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)

            if is_main_process():
                # Performance evaluation scatter plot
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.scatter(yhat, y)
                ax.set_title(f'R^2 = {r2:.3f} | MSE = {mse:.3f} | MAE: {mae:.3f}', fontsize=14)
                ax.set_xlabel(f'Predicted {task.task_name}', fontsize=13)
                ax.set_ylabel(f'True {task.task_name}', fontsize=13)
                fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_preds.png'), dpi=300, bbox_inches='tight')
                fig.clear()
                plt.close(fig)

        if is_main_process():
            # Save video-level and study-level predictions for task
            video_pred_df.to_csv(os.path.join(model_dir, 'preds', task.task_name, f'{split}_{task.task_name}_video_preds.csv'), index=False)
            study_pred_df.to_csv(os.path.join(model_dir, 'preds', task.task_name, f'{split}_{task.task_name}_preds.csv'), index=False)

        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    mean_classification_auroc, mean_classification_ap = np.nanmean(classification_aurocs), np.nanmean(classification_aps)
    mean_regression_r2, mean_regression_mse, mean_regression_mae = np.nanmean(regression_r2s), np.nanmean(regression_mses), np.nanmean(regression_maes)
    mean_auroc_r2 = (mean_classification_auroc + mean_regression_r2) / 2
    
    out_str += f'[OVERALL] Mean (AUROC, R^2): {mean_auroc_r2:.3f}'
    out_str += f'[CLASSIFICATION] Mean AUROC: {mean_classification_auroc:.3f} | Mean AP: {mean_classification_ap:.3f} ({classification_aurocs.size} total classification tasks)\n'
    out_str += f'[REGRESSION] Mean R^2: {mean_regression_r2:.3f} | Mean MSE: {mean_regression_mse:.3f} | Mean MAE: {mean_regression_mae:.3f} ({regression_r2s.size} total regression tasks)\n'

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')
        
    # Print overall summary text and save to text file
    print(out_str)
    if is_main_process():
        f = open(os.path.join(model_dir, f'{split}_summary.txt'), 'w')
        f.write(out_str)
        f.close()

        # Plot history of all metrics
        for col in np.setdiff1d(history.columns.values, ['epoch', 'phase']):
            task_name = col.split('_')[0]

            sub_history = history.dropna(subset=col).reset_index(drop=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(sub_history.loc[sub_history['phase'] == 'train', 'epoch'], sub_history.loc[sub_history['phase'] == 'train', col], marker='o', label='train')
            ax.plot(sub_history.loc[sub_history['phase'] == 'val', 'epoch'], sub_history.loc[sub_history['phase'] == 'val', col], marker='o', label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.legend()
            if task_name in [t.task_name for t in tasks]:
                fig.savefig(os.path.join(model_dir, 'history_plots', task_name, f'{col}_history.png'), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(model_dir, 'history_plots', f'{col}_history.png'), dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close(fig)

def evaluate_echonetdynamic(model, tasks, loss_fxns, data_loader, split, history, model_dir, weights, amp=False, plot_history=False):
    # Load weights determined by early stopping
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        msg = model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights)
    print(msg)
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                    task_data[task.task_name]['ys'].append(y.numpy(force=True))

                    # Collect auxiliary information
                    task_data[task.task_name]['fnames'].append(fname)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        y = y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss

                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    out_str = ''
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            out_str += f'--- {task.task_name} [{task.task_type}] (N=0) ---\n'
            continue

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        out_str += f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---\n'

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    fpr, tpr, _ = metrics.roc_curve(binary_y, yhat[:, class_idx])
                    prs, res, _ = metrics.precision_recall_curve(binary_y, yhat[:, class_idx])

                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                    # Plot class-specific ROC curve
                    roc_ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUROC: {auroc:.3f})')
                    # Plot class-specific PR curve
                    p = pr_ax.plot(res, prs, lw=2, label=f'{class_name} (AP: {ap:.3f})')
                    pr_ax.axhline(y=binary_y.sum()/binary_y.size, color=p[0].get_color(), lw=2, linestyle='--')

                    aurocs.append(auroc)
                    aps.append(ap)
                out_str += f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            out_str += f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

                fpr, tpr, _ = metrics.roc_curve(y, yhat)
                prs, res, _ = metrics.precision_recall_curve(y, yhat)

                # Plot class-specific ROC curve
                roc_ax.plot(fpr, tpr, lw=2, label=f'{task.class_names[1]} (AUROC: {auroc:.3f})')
                # Plot class-specific PR curve
                p = pr_ax.plot(res, prs, lw=2, label=f'{task.class_names[1]} (AP: {ap:.3f})')
                pr_ax.axhline(y=y.sum()/y.size, color=p[0].get_color(), lw=2, linestyle='--')
            
            out_str += f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(auroc)
            classification_aps.append(ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)
            
            out_str += f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}\n'

            # Keep track of overall mean classification metrics
            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)

            if is_main_process():
                # Performance evaluation scatter plot
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.scatter(yhat, y)
                ax.set_title(f'R^2 = {r2:.3f} | MSE = {mse:.3f} | MAE: {mae:.3f}', fontsize=14)
                ax.set_xlabel(f'Predicted {task.task_name}', fontsize=13)
                ax.set_ylabel(f'True {task.task_name}', fontsize=13)
                fig.savefig(os.path.join(model_dir, 'results_plots', f'echonet-dynamic_{split}_{task.task_name}_preds.png'), dpi=300, bbox_inches='tight')
                fig.clear()
                plt.close(fig)

        if is_main_process():
            # Save video-level and study-level predictions for task
            study_pred_df.to_csv(os.path.join(model_dir, 'preds', task.task_name, f'echonet-dynamic_{split}_{task.task_name}_preds.csv'), index=False)

        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    mean_classification_auroc, mean_classification_ap = np.nanmean(classification_aurocs), np.nanmean(classification_aps)
    mean_regression_r2, mean_regression_mse, mean_regression_mae = np.nanmean(regression_r2s), np.nanmean(regression_mses), np.nanmean(regression_maes)
    out_str += f'[CLASSIFICATION] Mean AUROC: {mean_classification_auroc:.3f} | Mean AP: {mean_classification_ap:.3f} ({classification_aurocs.size} total classification tasks)\n'
    out_str += f'[REGRESSION] Mean R^2: {mean_regression_r2:.3f} | Mean MSE: {mean_regression_mse:.3f} | Mean MAE: {mean_regression_mae:.3f} ({regression_r2s.size} total regression tasks)\n'

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')
        
    # Print overall summary text and save to text file
    print(out_str)
    if is_main_process():
        f = open(os.path.join(model_dir, f'echonet-dynamic_{split}_summary.txt'), 'w')
        f.write(out_str)
        f.close()

def train_echonetdynamic(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, amp=False):
    data_loader.sampler.set_epoch(epoch)  # for DDP
    
    model.train()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    for b, batch in pbar:
        x = batch['x'].cuda(memory_format=torch.channels_last_3d)
        fname = batch['fname']

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            # Forward pass
            out_dict = model(x)

            # Compute loss for each task
            losses = []
            for task in tasks:
                yhat = out_dict[task.task_name]
                y = batch[task.task_name].cuda()

                # Collect (masked) true and predicted labels
                if task.task_type == 'multi-class_classification':
                    task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                elif task.task_type == 'binary_classification':
                    task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                else:
                    task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                task_data[task.task_name]['ys'].append(y.numpy(force=True))

                # Collect (masked) auxiliary information
                task_data[task.task_name]['fnames'].append(fname)

                # For CrossEntropyLoss, target must have shape (N,)
                if task.task_type == 'multi-class_classification':
                    masked_y = y.squeeze(1)

                # Compute task loss
                loss = loss_fxns[task.task_name](yhat, y)
                # Scale down regression loss based on mean value
                if task.task_type == 'regression':
                    loss /= task.mean
                losses.append(loss)

                # Keep track of task losses for each batch
                task_data[task.task_name]['losses'].append(loss.item())
                del loss
            # Compute overall loss
            if len(losses) == 0:
                continue
            else:
                loss = sum(losses) / len(losses)
                del losses
                
                # Keep running sum of losses for each batch
                running_loss += loss.item()
            overall_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dict

    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.array(overall_losses).ravel()

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'train'
    current_metrics_dict['loss'] = overall_losses.mean()
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
        
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    gc.collect()

    return pd.concat([history, current_metrics], axis=0)

def validate_echonetdynamic(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp=False, scheduler=None):
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                    task_data[task.task_name]['ys'].append(y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    task_data[task.task_name]['fnames'].append(fname)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.concatenate(all_gather(overall_losses)).ravel()
    val_loss = overall_losses.mean()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'val'
    current_metrics_dict['loss'] = val_loss
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        # task_loss = np.concatenate(all_gather(task_data[task.task_name]['losses'])).mean()
        task_loss = np.array(task_data[task.task_name]['losses']).mean()  # mean reduce
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
                    
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping. Save model weights only when val loss has improved
    if val_loss < early_stopping_dict['best_loss']:
        print(f'EARLY STOPPING: Loss has improved from {early_stopping_dict["best_loss"]:.3f} to {val_loss:.3f}! Saving weights.')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_loss'] = val_loss
        best_model_wts = deepcopy(model.module.state_dict()) if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel) else deepcopy(model.state_dict())
        if is_main_process():
            torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'EARLY STOPPING: Loss has not improved from {early_stopping_dict["best_loss"]:.3f}')
        early_stopping_dict['epochs_no_improve'] += 1
    dist.barrier()

    # Apply learning rate scheduler (if given)
    if scheduler is not None:
        scheduler.step()

    history = pd.concat([history, current_metrics], axis=0)
    # Plot current metrics
    if is_main_process():
        for col in np.setdiff1d(history.columns.values, ['epoch', 'phase']):
            task_name = col.split('_')[0]

            sub_history = history.dropna(subset=col).reset_index(drop=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(sub_history.loc[sub_history['phase'] == 'train', 'epoch'], sub_history.loc[sub_history['phase'] == 'train', col], marker='o', label='train')
            ax.plot(sub_history.loc[sub_history['phase'] == 'val', 'epoch'], sub_history.loc[sub_history['phase'] == 'val', col], marker='o', label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.legend()
            if task_name in [t.task_name for t in tasks]:
                fig.savefig(os.path.join(model_dir, 'history_plots', task_name, f'{col}_history.png'), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(model_dir, 'history_plots', f'{col}_history.png'), dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close(fig)

    gc.collect()

    return history, early_stopping_dict, best_model_wts

def evaluate_echonetpediatric(model, tasks, loss_fxns, data_loader, split, fold, history, model_dir, weights, amp=False, plot_history=False):
    # Load weights determined by early stopping
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        msg = model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights)
    print(msg)
    model.eval()

    if fold == '':
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    else:
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'(FOLD {fold+1}) [{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'fnames': [], 'views': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']
            acc_num = batch['acc_num']
            view = batch['view']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()

                    # Collect true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                    task_data[task.task_name]['ys'].append(y.numpy(force=True))

                    # Collect auxiliary information
                    task_data[task.task_name]['fnames'].append(fname)
                    task_data[task.task_name]['acc_nums'].append(acc_num)
                    task_data[task.task_name]['views'].append(view)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        y = y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())

                    print(task.task_name, task.mean, loss.item())
                    for y_, yh_ in zip(y.numpy(force=True), yhat.numpy(force=True)):
                        print(y_, yh_)

                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    out_str = ''
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            out_str += f'--- {task.task_name} [{task.task_type}] (N=0) ---\n'
            continue

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
            'view': np.concatenate(task_data[task.task_name]['views'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        out_str += f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---\n'

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    fpr, tpr, _ = metrics.roc_curve(binary_y, yhat[:, class_idx])
                    prs, res, _ = metrics.precision_recall_curve(binary_y, yhat[:, class_idx])

                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                    # Plot class-specific ROC curve
                    roc_ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUROC: {auroc:.3f})')
                    # Plot class-specific PR curve
                    p = pr_ax.plot(res, prs, lw=2, label=f'{class_name} (AP: {ap:.3f})')
                    pr_ax.axhline(y=binary_y.sum()/binary_y.size, color=p[0].get_color(), lw=2, linestyle='--')

                    aurocs.append(auroc)
                    aps.append(ap)
                out_str += f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            out_str += f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)

        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

                fpr, tpr, _ = metrics.roc_curve(y, yhat)
                prs, res, _ = metrics.precision_recall_curve(y, yhat)

                # Plot class-specific ROC curve
                roc_ax.plot(fpr, tpr, lw=2, label=f'{task.class_names[1]} (AUROC: {auroc:.3f})')
                # Plot class-specific PR curve
                p = pr_ax.plot(res, prs, lw=2, label=f'{task.class_names[1]} (AP: {ap:.3f})')
                pr_ax.axhline(y=y.sum()/y.size, color=p[0].get_color(), lw=2, linestyle='--')
            
            out_str += f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(auroc)
            classification_aps.append(ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)
            
            out_str += f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}\n'

            # Keep track of overall mean classification metrics
            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)

            if is_main_process():
                # Performance evaluation scatter plot
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.scatter(yhat, y)
                ax.set_title(f'R^2 = {r2:.3f} | MSE = {mse:.3f} | MAE: {mae:.3f}', fontsize=14)
                ax.set_xlabel(f'Predicted {task.task_name}', fontsize=13)
                ax.set_ylabel(f'True {task.task_name}', fontsize=13)
                if fold == '':
                    fig.savefig(os.path.join(model_dir, f'echonet-pediatric_{split}_{task.task_name}_preds.png'), dpi=300, bbox_inches='tight')
                else:
                    fig.savefig(os.path.join(model_dir, f'fold_{fold}', 'results_plots', f'echonet-pediatric_{split}_{task.task_name}_preds.png'), dpi=300, bbox_inches='tight')
                fig.clear()
                plt.close(fig)

        if is_main_process():
            if fold == '':
                # Save video-level and study-level predictions for task
                video_pred_df.to_csv(os.path.join(model_dir, f'echonet-pediatric_{split}_{task.task_name}_video_preds.csv'), index=False)
                study_pred_df.to_csv(os.path.join(model_dir, f'echonet-pediatric_{split}_{task.task_name}_preds.csv'), index=False)
            else:
                # Save video-level and study-level predictions for task
                video_pred_df.to_csv(os.path.join(model_dir, f'fold_{fold}', 'preds', task.task_name, f'echonet-pediatric_{split}_{task.task_name}_video_preds.csv'), index=False)
                study_pred_df.to_csv(os.path.join(model_dir, f'fold_{fold}', 'preds', task.task_name, f'echonet-pediatric_{split}_{task.task_name}_preds.csv'), index=False)

        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    mean_classification_auroc, mean_classification_ap = np.nanmean(classification_aurocs), np.nanmean(classification_aps)
    mean_regression_r2, mean_regression_mse, mean_regression_mae = np.nanmean(regression_r2s), np.nanmean(regression_mses), np.nanmean(regression_maes)
    out_str += f'[CLASSIFICATION] Mean AUROC: {mean_classification_auroc:.3f} | Mean AP: {mean_classification_ap:.3f} ({classification_aurocs.size} total classification tasks)\n'
    out_str += f'[REGRESSION] Mean R^2: {mean_regression_r2:.3f} | Mean MSE: {mean_regression_mse:.3f} | Mean MAE: {mean_regression_mae:.3f} ({regression_r2s.size} total regression tasks)\n'

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')
        
    # Print overall summary text and save to text file
    print(out_str)
    if is_main_process():
        if fold == '':
            f = open(os.path.join(model_dir, f'echonet-pediatric_{split}_summary.txt'), 'w')
        else:
            f = open(os.path.join(model_dir, f'fold_{fold}', f'echonet-pediatric_{split}_summary.txt'), 'w')
        f.write(out_str)
        f.close()

def train_echonetpediatric(model, tasks, fold, loss_fxns, optimizer, data_loader, history, epoch, model_dir, amp=False):
    data_loader.sampler.set_epoch(epoch)  # for DDP
    
    model.train()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'(FOLD {fold+1}) Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'invalid_batches': 0} for task in tasks}
    for b, batch in pbar:
        x = batch['x'].cuda(memory_format=torch.channels_last_3d)
        acc_num = batch['acc_num']

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            # Forward pass
            out_dict = model(x)

            # Compute loss for each task
            losses = []
            for task in tasks:
                yhat = out_dict[task.task_name]
                y = batch[task.task_name].cuda()

                # Collect true and predicted labels
                if task.task_type == 'multi-class_classification':
                    task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                elif task.task_type == 'binary_classification':
                    task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                else:
                    task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                task_data[task.task_name]['ys'].append(y.numpy(force=True))

                # Collect auxiliary information
                task_data[task.task_name]['acc_nums'].append(acc_num)

                # For CrossEntropyLoss, target must have shape (N,)
                if task.task_type == 'multi-class_classification':
                    masked_y = y.squeeze(1)

                # Compute task loss
                loss = loss_fxns[task.task_name](yhat, y)
                # Scale down regression loss based on mean value
                if task.task_type == 'regression':
                    loss /= task.mean
                losses.append(loss)

                # Keep track of task losses for each batch
                task_data[task.task_name]['losses'].append(loss.item())
                del loss
            # Compute overall loss
            if len(losses) == 0:
                continue
            else:
                loss = sum(losses) / len(losses)
                del losses
                
                # Keep running sum of losses for each batch
                running_loss += loss.item()
            overall_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dict

    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.array(overall_losses).ravel()

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'train'
    current_metrics_dict['loss'] = overall_losses.mean()
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
        
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, f'fold_{fold}', 'history.csv'), mode='a', header=False, index=False)

    gc.collect()

    return pd.concat([history, current_metrics], axis=0)

def validate_echonetpediatric(model, tasks, fold, loss_fxns, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp=False, scheduler=None):
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'(FOLD {fold+1}) [VAL] Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'acc_nums': [], 'fnames': [], 'views': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']
            acc_num = batch['acc_num']
            view = batch['view']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()

                    # Collect true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(yhat.numpy(force=True))
                    task_data[task.task_name]['ys'].append(y.numpy(force=True))

                    # Collect auxiliary information
                    task_data[task.task_name]['fnames'].append(fname)
                    task_data[task.task_name]['acc_nums'].append(acc_num)
                    task_data[task.task_name]['views'].append(view)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        y = y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.concatenate(all_gather(overall_losses)).ravel()
    val_loss = overall_losses.mean()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'val'
    current_metrics_dict['loss'] = val_loss
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        # task_loss = np.concatenate(all_gather(task_data[task.task_name]['losses'])).mean()
        task_loss = np.array(task_data[task.task_name]['losses']).mean()  # mean reduce
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        video_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
            'acc_num': np.concatenate(task_data[task.task_name]['acc_nums'], axis=0),
            'view': np.concatenate(task_data[task.task_name]['views'], axis=0),
        })

        # Aggregate video-level predictions into study-level predictions by averaging
        study_pred_df = video_pred_df.groupby('acc_num', as_index=False).agg({'y': np.mean, 'yhat': np.mean})

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
                    
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, f'fold_{fold}', 'history.csv'), mode='a', header=False, index=False)

    # Early stopping. Save model weights only when val loss has improved
    if val_loss < early_stopping_dict['best_loss']:
        print(f'EARLY STOPPING: Loss has improved from {early_stopping_dict["best_loss"]:.3f} to {val_loss:.3f}! Saving weights.')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_loss'] = val_loss
        best_model_wts = deepcopy(model.module.state_dict()) if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel) else deepcopy(model.state_dict())
        if is_main_process():
            torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'fold_{fold}', f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'EARLY STOPPING: Loss has not improved from {early_stopping_dict["best_loss"]:.3f}')
        early_stopping_dict['epochs_no_improve'] += 1
    dist.barrier()

    # Apply learning rate scheduler (if given)
    if scheduler is not None:
        scheduler.step()

    history = pd.concat([history, current_metrics], axis=0)
    # Plot current metrics
    if is_main_process():
        for col in np.setdiff1d(history.columns.values, ['epoch', 'phase']):
            task_name = col.split('_')[0]

            sub_history = history.dropna(subset=col).reset_index(drop=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(sub_history.loc[sub_history['phase'] == 'train', 'epoch'], sub_history.loc[sub_history['phase'] == 'train', col], marker='o', label='train')
            ax.plot(sub_history.loc[sub_history['phase'] == 'val', 'epoch'], sub_history.loc[sub_history['phase'] == 'val', col], marker='o', label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.legend()
            if task_name in [t.task_name for t in tasks]:
                fig.savefig(os.path.join(model_dir, f'fold_{fold}', 'history_plots', task_name, f'{col}_history.png'), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(model_dir, f'fold_{fold}', 'history_plots', f'{col}_history.png'), dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close(fig)

    gc.collect()

    return history, early_stopping_dict, best_model_wts

def evaluate_echonetlvh(model, tasks, loss_fxns, data_loader, split, history, model_dir, weights, amp=False, plot_history=False):
    # Load weights determined by early stopping
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        msg = model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights)
    print(msg)
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()

                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_fname = [f for f, m in zip(fname, mask) if m]
                    task_data[task.task_name]['fnames'].append(masked_fname)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())

                    print(task.task_name, task.mean, loss.item())
                    for y_, yh_ in zip(y.numpy(force=True), yhat.numpy(force=True)):
                        print(y_, yh_)

                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    out_str = ''
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            out_str += f'--- {task.task_name} [{task.task_type}] (N=0) ---\n'
            continue

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        out_str += f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---\n'

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    fpr, tpr, _ = metrics.roc_curve(binary_y, yhat[:, class_idx])
                    prs, res, _ = metrics.precision_recall_curve(binary_y, yhat[:, class_idx])

                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                    # Plot class-specific ROC curve
                    roc_ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUROC: {auroc:.3f})')
                    # Plot class-specific PR curve
                    p = pr_ax.plot(res, prs, lw=2, label=f'{class_name} (AP: {ap:.3f})')
                    pr_ax.axhline(y=binary_y.sum()/binary_y.size, color=p[0].get_color(), lw=2, linestyle='--')

                    aurocs.append(auroc)
                    aps.append(ap)
                out_str += f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            out_str += f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)

        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Initialize performance summary plots
            roc_fig, roc_ax = plt.subplots(1, 1, figsize=(6, 6))
            pr_fig, pr_ax = plt.subplots(1, 1, figsize=(6, 6))

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

                fpr, tpr, _ = metrics.roc_curve(y, yhat)
                prs, res, _ = metrics.precision_recall_curve(y, yhat)

                # Plot class-specific ROC curve
                roc_ax.plot(fpr, tpr, lw=2, label=f'{task.class_names[1]} (AUROC: {auroc:.3f})')
                # Plot class-specific PR curve
                p = pr_ax.plot(res, prs, lw=2, label=f'{task.class_names[1]} (AP: {ap:.3f})')
                pr_ax.axhline(y=y.sum()/y.size, color=p[0].get_color(), lw=2, linestyle='--')
            
            out_str += f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}\n'

            # Keep track of overall mean classification metrics
            classification_aurocs.append(auroc)
            classification_aps.append(ap)

            if is_main_process():
                # Save ROC plot
                roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                roc_ax.set_xlim([-0.05, 1.0])
                roc_ax.set_ylim([0.0, 1.05])
                roc_ax.set_xlabel('1 - Specificity', fontsize=13)
                roc_ax.set_ylabel('Sensitivity', fontsize=13)
                roc_ax.legend(loc="lower right", fontsize=11)
                roc_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_roc.png'), dpi=300, bbox_inches='tight')
                roc_fig.clear()
                plt.close(roc_fig)

                # Save PR plot
                pr_ax.set_xlim([-0.05, 1.05])
                pr_ax.set_ylim([-0.05, 1.05])
                pr_ax.set_xlabel('Recall', fontsize=13)
                pr_ax.set_ylabel('Precision', fontsize=13)
                pr_ax.legend(loc="upper right", fontsize=11)
                pr_fig.savefig(os.path.join(model_dir, 'results_plots', task.task_name, f'{split}_{task.task_name}_pr.png'), dpi=300, bbox_inches='tight')
                pr_fig.clear()
                plt.close(pr_fig)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)
            
            out_str += f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}\n'

            # Keep track of overall mean classification metrics
            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)

            if is_main_process():
                # Performance evaluation scatter plot
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.scatter(yhat, y)
                ax.set_title(f'R^2 = {r2:.3f} | MSE = {mse:.3f} | MAE: {mae:.3f}', fontsize=14)
                ax.set_xlabel(f'Predicted {task.task_name}', fontsize=13)
                ax.set_ylabel(f'True {task.task_name}', fontsize=13)
                fig.savefig(os.path.join(model_dir, 'results_plots', f'echonet-lvh_{split}_{task.task_name}_preds.png'), dpi=300, bbox_inches='tight')
                fig.clear()
                plt.close(fig)

        if is_main_process():
            # Save video-level and study-level predictions for task
            study_pred_df.to_csv(os.path.join(model_dir, 'preds', task.task_name, f'echonet-lvh_{split}_{task.task_name}_preds.csv'), index=False)

        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    mean_classification_auroc, mean_classification_ap = np.nanmean(classification_aurocs), np.nanmean(classification_aps)
    mean_regression_r2, mean_regression_mse, mean_regression_mae = np.nanmean(regression_r2s), np.nanmean(regression_mses), np.nanmean(regression_maes)
    out_str += f'[CLASSIFICATION] Mean AUROC: {mean_classification_auroc:.3f} | Mean AP: {mean_classification_ap:.3f} ({classification_aurocs.size} total classification tasks)\n'
    out_str += f'[REGRESSION] Mean R^2: {mean_regression_r2:.3f} | Mean MSE: {mean_regression_mse:.3f} | Mean MAE: {mean_regression_mae:.3f} ({regression_r2s.size} total regression tasks)\n'

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')
        
    # Print overall summary text and save to text file
    print(out_str)
    if is_main_process():
        f = open(os.path.join(model_dir, f'echonet-lvh_{split}_summary.txt'), 'w')
        f.write(out_str)
        f.close()

def train_echonetlvh(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, amp=False):
    data_loader.sampler.set_epoch(epoch)  # for DDP
    
    model.train()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    for b, batch in pbar:
        x = batch['x'].cuda(memory_format=torch.channels_last_3d)
        fname = batch['fname']

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            # Forward pass
            out_dict = model(x)

            # Compute loss for each task
            losses = []
            for task in tasks:
                yhat = out_dict[task.task_name]
                y = batch[task.task_name].cuda()
                mask = batch[task.task_name+'_mask'].cuda()

                # If batch contains *only* missing values for task, then skip loss computation
                if mask.sum() == 0:
                    task_data[task.task_name]['invalid_batches'] += 1
                    continue

                # Mask out missing values from loss computation
                masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                # Collect (masked) true and predicted labels
                if task.task_type == 'multi-class_classification':
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                elif task.task_type == 'binary_classification':
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                else:
                    task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                # Collect (masked) auxiliary information
                masked_fname = [f for f, m in zip(fname, mask) if m]
                task_data[task.task_name]['fnames'].append(masked_fname)

                # For CrossEntropyLoss, target must have shape (N,)
                if task.task_type == 'multi-class_classification':
                    masked_y = masked_y.squeeze(1)

                # Compute task loss
                loss = loss_fxns[task.task_name](yhat, y)
                # Scale down regression loss based on mean value
                if task.task_type == 'regression':
                    loss /= task.mean
                losses.append(loss)

                # Keep track of task losses for each batch
                task_data[task.task_name]['losses'].append(loss.item())
            # Compute overall loss
            if len(losses) == 0:
                continue
            else:
                loss = sum(losses) / len(losses)
                del losses
                
                # Keep running sum of losses for each batch
                running_loss += loss.item()
            overall_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dict

    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.array(overall_losses).ravel()

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'train'
    current_metrics_dict['loss'] = overall_losses.mean()
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
        
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    gc.collect()

    return pd.concat([history, current_metrics], axis=0)

def validate_echonetlvh(model, tasks, loss_fxns, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp=False, scheduler=None):
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.

    overall_losses = []
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'fnames': [], 'invalid_batches': 0} for task in tasks}
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x']
            fname = batch['fname']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass
                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()

                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])

                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.float().numpy(force=True))
                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_fname = [f for f, m in zip(fname, mask) if m]
                    task_data[task.task_name]['fnames'].append(masked_fname)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](yhat, y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss
                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

    del x, out_dicts
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    if is_main_process():
        s = time.perf_counter()

    # Gather losses across processes and flatten
    overall_losses = np.concatenate(all_gather(overall_losses)).ravel()
    val_loss = overall_losses.mean()

    # Gather task data (dict) across processes and merge by concatenating lists from shared keys together
    task_data = merge_task_dicts(all_gather(task_data))

    # Compute and log metrics for each task
    classification_aurocs, classification_aps = [], []
    regression_r2s, regression_mses, regression_maes = [], [], []
    current_metrics_dict = {k: np.nan for k in history.columns.values}
    current_metrics_dict['epoch'] = epoch
    current_metrics_dict['phase'] = 'val'
    current_metrics_dict['loss'] = val_loss
    for task in tasks:
        # Compute task loss (accounting for invalid batches)
        task_data[task.task_name]['invalid_batches'] = np.array(task_data[task.task_name]['invalid_batches']).sum()  # sum reduce (currently list from each process)
        valid_batches = b-task_data[task.task_name]['invalid_batches']+1

        if valid_batches == 0:
            current_metrics_dict[f'{task.task_name}_loss'] = np.nan

            if task.task_type == 'multi-class_classification':
                current_metrics_dict[f'{task.task_name}_mean_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_mean_ap'] = np.nan
                for class_name in task.class_names:
                    current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = np.nan
                    current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = np.nan
            elif task.task_type == 'binary_classification':
                current_metrics_dict[f'{task.task_name}_auroc'] = np.nan
                current_metrics_dict[f'{task.task_name}_ap'] = np.nan
            else:
                current_metrics_dict[f'{task.task_name}_r2'] = np.nan
                current_metrics_dict[f'{task.task_name}_mse'] = np.nan
                current_metrics_dict[f'{task.task_name}_mae'] = np.nan
            continue

        task_loss = np.array(task_data[task.task_name]['losses']).mean()  # mean reduce
        current_metrics_dict[f'{task.task_name}_loss'] = task_loss

        study_pred_df = pd.DataFrame({
            'y': np.concatenate(task_data[task.task_name]['ys'], axis=0).ravel(),
            'yhat': [x for x in np.concatenate(task_data[task.task_name]['yhats'], axis=0)],
            'fname': np.concatenate(task_data[task.task_name]['fnames'], axis=0),
        })

        print(f'--- {task.task_name} [{task.task_type}] (N={study_pred_df.shape[0]}) ---')

        if task.task_type == 'multi-class_classification':
            y = study_pred_df['y'].values
            yhat = np.stack(study_pred_df['yhat'].values, axis=0)  # (N,C)

            # Compute classification metrics for each class individually
            aurocs, aps = [], []
            for class_idx, class_name in zip(task.class_indices, task.class_names):
                binary_y = (y == class_idx).astype(int)

                if binary_y.sum() in [0, binary_y.size]:  # if all one class, cannot compute metric
                    auroc, ap = np.nan, np.nan
                else:
                    auroc = metrics.roc_auc_score(binary_y, yhat[:, class_idx])
                    ap = metrics.average_precision_score(binary_y, yhat[:, class_idx])

                current_metrics_dict[f'{task.task_name}_{class_name}_auroc'] = auroc
                current_metrics_dict[f'{task.task_name}_{class_name}_ap'] = ap
                aurocs.append(auroc)
                aps.append(ap)
                
                print(f'\t[{class_name.upper()}] AUROC: {auroc:.3f} | AP: {ap:.3f}')
            mean_auroc, mean_ap = np.nanmean(aurocs), np.nanmean(aps)
            current_metrics_dict[f'{task.task_name}_mean_auroc'] = mean_auroc
            current_metrics_dict[f'{task.task_name}_mean_ap'] = mean_ap
            print(f'\t[MEAN] AUROC: {mean_auroc:.3f} | AP: {mean_ap:.3f}')

            classification_aurocs.append(mean_auroc)
            classification_aps.append(mean_ap)
        elif task.task_type == 'binary_classification':
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute binary classification metrics
            if y.sum() in [0, y.size]:  # if all one class, cannot compute metric
                auroc, ap = np.nan, np.nan
            else:
                auroc = metrics.roc_auc_score(y, yhat)
                ap = metrics.average_precision_score(y, yhat)

            current_metrics_dict[f'{task.task_name}_auroc'] = auroc
            current_metrics_dict[f'{task.task_name}_ap'] = ap
            
            print(f'\tAUROC: {auroc:.3f} | AP: {ap:.3f}')

            classification_aurocs.append(auroc)
            classification_aps.append(ap)
        else:
            y = study_pred_df['y'].values
            yhat = study_pred_df['yhat'].values

            # Compute regression metrics
            r2 = metrics.r2_score(y, yhat)
            mse = metrics.mean_squared_error(y, yhat)
            mae = metrics.mean_absolute_error(y, yhat)

            current_metrics_dict[f'{task.task_name}_r2'] = r2
            current_metrics_dict[f'{task.task_name}_mse'] = mse
            current_metrics_dict[f'{task.task_name}_mae'] = mae

            print(f'\tR^2: {r2:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f}')

            regression_r2s.append(r2)
            regression_mses.append(mse)
            regression_maes.append(mae)
                    
        # Free task-specific data from memory as it is processed
        del task_data[task.task_name]

    # Overall mean classification and regression metrics for each task type
    classification_aurocs, classification_aps = np.array(classification_aurocs), np.array(classification_aps)
    regression_r2s, regression_mses, regression_maes = np.array(regression_r2s), np.array(regression_mses), np.array(regression_maes)
    current_metrics_dict['mean_classification_auroc'] = np.nanmean(classification_aurocs)
    current_metrics_dict['mean_classification_ap'] = np.nanmean(classification_aps)
    current_metrics_dict['mean_regression_r2'] = np.nanmean(regression_r2s)
    current_metrics_dict['mean_regression_mse'] = np.nanmean(regression_mses)
    current_metrics_dict['mean_regression_mae'] = np.nanmean(regression_maes)
    print(f'[CLASSIFICATION] Mean AUROC: {current_metrics_dict["mean_classification_auroc"]:.3f} | Mean AP: {current_metrics_dict["mean_classification_ap"]:.3f} ({classification_aurocs[~np.isnan(classification_aurocs)].size} total classification tasks)')
    print(f'[REGRESSION] Mean R^2: {current_metrics_dict["mean_regression_r2"]:.3f} | Mean MSE: {current_metrics_dict["mean_regression_mse"]:.3f} | Mean MAE: {current_metrics_dict["mean_regression_mae"]:.3f} ({regression_r2s[~np.isnan(regression_r2s)].size} total regression tasks)')

    if is_main_process():
        e = time.perf_counter()
        print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    # Convert dict of current metrics to pandas DataFrame and append to "history"
    current_metrics = pd.DataFrame([current_metrics_dict])
    if is_main_process():
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping. Save model weights only when val loss has improved
    if val_loss < early_stopping_dict['best_loss']:
        print(f'EARLY STOPPING: Loss has improved from {early_stopping_dict["best_loss"]:.3f} to {val_loss:.3f}! Saving weights.')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_loss'] = val_loss
        best_model_wts = deepcopy(model.module.state_dict()) if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel) else deepcopy(model.state_dict())
        if is_main_process():
            torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'EARLY STOPPING: Loss has not improved from {early_stopping_dict["best_loss"]:.3f}')
        early_stopping_dict['epochs_no_improve'] += 1
    dist.barrier()

    # Apply learning rate scheduler (if given)
    if scheduler is not None:
        scheduler.step()

    history = pd.concat([history, current_metrics], axis=0)
    # Plot current metrics
    if is_main_process():
        for col in np.setdiff1d(history.columns.values, ['epoch', 'phase']):
            task_name = col.split('_')[0]

            sub_history = history.dropna(subset=col).reset_index(drop=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(sub_history.loc[sub_history['phase'] == 'train', 'epoch'], sub_history.loc[sub_history['phase'] == 'train', col], marker='o', label='train')
            ax.plot(sub_history.loc[sub_history['phase'] == 'val', 'epoch'], sub_history.loc[sub_history['phase'] == 'val', col], marker='o', label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.legend()
            if task_name in [t.task_name for t in tasks]:
                fig.savefig(os.path.join(model_dir, 'history_plots', task_name, f'{col}_history.png'), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(model_dir, 'history_plots', f'{col}_history.png'), dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close(fig)

    gc.collect()

    return history, early_stopping_dict, best_model_wts
