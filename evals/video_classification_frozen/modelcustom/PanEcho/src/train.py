import gc
import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.distributed as dist

from sklearn.utils import compute_class_weight

from dataset import EchoDataset
from ddp_utils import is_main_process, setup_for_distributed
from models import FrameTransformer, MultiTaskModel
from utils import Task, set_seed, worker_init_fn, val_worker_init_fn, train, validate, evaluate

def init_distributed():
    dist_url = 'env://'  # default

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # Make .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # Synchronize processes before proceeding
    dist.barrier()
    setup_for_distributed(rank == 0)
    
def main(args):
    # Create model name
    MODEL_NAME = f'{args.model_name}'
    MODEL_NAME += f'_drp-{args.transformer_dropout}' if args.transformer_dropout > 0 else ''
    MODEL_NAME += f'_{args.arch}' if args.arch != '' else ''
    MODEL_NAME += f'_{args.pooling}-pooling' if args.model_name == 'frame_transformer' else ''
    MODEL_NAME += '_pretr' if not args.rand_init else '_rand'
    MODEL_NAME += f'_{args.normalization}-norm' if args.normalization != '' else ''
    MODEL_NAME += '_aug' if args.augment else ''
    MODEL_NAME += f'_clip-len-{args.clip_len}'
    MODEL_NAME += f'_num-clips-{args.num_clips}'
    MODEL_NAME += '_cw' if args.use_class_weights else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_cos-anneal_T0-{args.T_0}_etamin-{args.eta_min}' if args.cos_anneal else ''
    MODEL_NAME += f'_{args.max_epochs}ep'
    MODEL_NAME += f'_patience-{args.patience}' if args.patience != 1e4 else ''
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'_adamw' if args.adamw else ''
    MODEL_NAME += f'_wd-{args.wd}' if args.wd != 0. else ''
    MODEL_NAME += f'_drp-{args.fc_dropout}'
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''
    MODEL_NAME += '_amp' if args.amp else ''

    # Set up model directory
    model_dir = os.path.join(args.output_dir, MODEL_NAME)
    if is_main_process():
        if not args.resume:
            # Create output directory for model (and delete if already exists)
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
            os.mkdir(model_dir)
            os.mkdir(os.path.join(model_dir, 'history_plots'))
            os.mkdir(os.path.join(model_dir, 'results_plots'))
            os.mkdir(os.path.join(model_dir, 'preds'))

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Get task information and labels
    train_df = pd.read_csv(os.path.join(args.data_dir, '041824_train_labels.csv'))
    val_df = pd.read_csv(os.path.join(args.data_dir, '041824_val_labels.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, '041824_test_labels.csv'))

    tasks = np.load(os.path.join(args.data_dir, '041824_tasks.npy'), allow_pickle=True)
    tasks = [t for t in tasks if t.task_name not in ['HCM-MRI-confirmed-or-suggestive', 'Amyloid']]

    # Create datasets
    train_dataset    = EchoDataset(data_dir=args.data_dir, data_df=train_df, tasks=tasks, split='train', clip_len=args.clip_len, num_clips=args.num_clips, augment=args.augment, normalization=args.normalization, train=True)
    val_dataset      = EchoDataset(data_dir=args.data_dir, data_df=val_df, tasks=tasks, split='val', clip_len=args.clip_len, num_clips=args.num_clips, normalization=args.normalization, train=False)
    test_dataset     = EchoDataset(data_dir=args.data_dir, data_df=test_df, tasks=tasks, split='test', clip_len=args.clip_len, num_clips=args.num_clips, normalization=args.normalization, train=False)

    # Create loaders
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True), num_workers=16, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset=val_dataset, shuffle=False), num_workers=16, worker_init_fn=val_worker_init_fn, persistent_workers=True)
    test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset=test_dataset, shuffle=False), num_workers=8, worker_init_fn=val_worker_init_fn, persistent_workers=True)

    # Create csv documenting training history
    columns = ['epoch', 'phase', 'loss', 'mean_auroc_r2', 'mean_classification_auroc', 'mean_classification_ap', 'mean_regression_r2', 'mean_regression_mse', 'mean_regression_mae']
    for task in tasks:
        name = task.task_name
        columns.append(f'{name}_loss')

        if task.task_type == 'regression':
            columns.append(f'{name}_r2')
            columns.append(f'{name}_mse')
            columns.append(f'{name}_mae')
        elif task.task_type == 'multi-class_classification':
            columns.append(f'{name}_mean_auroc')
            columns.append(f'{name}_mean_ap')
            for class_name in task.class_names:
                columns.append(f'{name}_{class_name}_auroc')
                columns.append(f'{name}_{class_name}_ap')
        else:
            columns.append(f'{name}_auroc')
            columns.append(f'{name}_ap')

        if is_main_process():
            if not args.resume:
                # Create subdirectories for each task
                os.mkdir(os.path.join(model_dir, 'history_plots', name))
                os.mkdir(os.path.join(model_dir, 'results_plots', name))
                os.mkdir(os.path.join(model_dir, 'preds', name))
    if not args.resume:
        history = pd.DataFrame(columns=columns)
        if is_main_process():
            history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Initialize encoder
    if args.model_name == '3dresnet18':
        encoder = torchvision.models.video.r3d_18(pretrained=not args.rand_init)
        encoder_dim = encoder.fc.in_features
        encoder.fc = torch.nn.Identity()
    elif args.model_name == 'frame_transformer':
        encoder = FrameTransformer(arch=args.arch, n_heads=args.n_heads, n_layers=args.n_layers, transformer_dropout=args.transformer_dropout, pooling=args.pooling, clip_len=args.clip_len)
        encoder_dim = encoder.encoder.n_features

    # Initialize multi-task model
    model = MultiTaskModel(encoder=encoder, encoder_dim=encoder_dim, tasks=train_dataset.tasks, fc_dropout=args.fc_dropout)  # important to pass training set task info (with mean values computed from training data)
    print(model)

    # Load weights if resuming training
    if args.resume:
        chkpts = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        idx = np.argmax([int(f.split('.')[0].split('-')[-1]) for f in chkpts])
        chkpt = torch.load(os.path.join(model_dir, chkpts[idx]), map_location='cpu')

        msg = model.load_state_dict(chkpt['weights'], strict=True)
        print('Loading best weights...')
        print(msg)

    # Convert BatchNorm to SyncBatchNorm for DDP training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Transfer model to GPU and convert to DDP
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], find_unused_parameters=True)

    # Define loss functions (MSE for regression tasks and cross-entropy for classification tasks)
    if args.use_class_weights:
        loss_fxns = {}
        for task in tasks:
            if task.task_type == 'regression':
                loss_fxns[task.task_name] = torch.nn.MSELoss()
            else:
                class_weights = compute_class_weight(class_weight='balanced', classes=np.sort(train_dataset.data_df[task.task_name].dropna().unique()), y=train_dataset.data_df[task.task_name].dropna().values)

                if task.task_type == 'multi-class_classification':
                    if class_weights.size == 1:
                        loss_fxns[task.task_name] = torch.nn.CrossEntropyLoss()
                    else:
                        class_weights_tensor = torch.Tensor(class_weights).cuda()
                        loss_fxns[task.task_name] = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                else:
                    if class_weights.size == 1:
                        loss_fxns[task.task_name] = torch.nn.BCEWithLogitsLoss()
                    else:
                        class_weights_tensor = torch.Tensor([(class_weights / class_weights.min())[1]]).cuda()
                        loss_fxns[task.task_name] = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    else:
        loss_fxns = {}
        for task in tasks:
            if task.task_type == 'multi-class_classification':
                loss_fxns[task.task_name] = torch.nn.CrossEntropyLoss()
            elif task.task_type == 'binary_classification':
                loss_fxns[task.task_name] = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fxns[task.task_name] = torch.nn.MSELoss()

    # Initialize optimizer
    if args.adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Initialize learning rate scheduler
    if args.cos_anneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
    elif args.reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max', verbose=True)
    else:
        scheduler = None

    # Prepare to train with early stopping based on mean of AUROC across classification tasks and R^2 across regression tasks
    epoch = 1
    early_stopping_dict = {'best_mean_auroc_r2': 0., 'epochs_no_improve': 0}
    best_model_wts = None

    # If resuming training, load optimizer and training history from last available epoch
    if args.resume:
        history = pd.read_csv(os.path.join(model_dir, 'history.csv'))

        optimizer.load_state_dict(chkpt['optimizer'])
        epoch = int(np.max([int(f.split('.')[0].split('-')[-1]) for f in chkpts]))+1

        best_model_wts = chkpt['weights']
        del chkpt

    # Training loop
    while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] < args.patience:
        history = train(model=model, tasks=tasks, loss_fxns=loss_fxns, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, amp=args.amp)
        history, early_stopping_dict, best_model_wts = validate(model=model, tasks=tasks, loss_fxns=loss_fxns, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, amp=args.amp, scheduler=scheduler)

        epoch += 1
    
    # Evaluate on validation set
    evaluate(model=model, tasks=tasks, loss_fxns=loss_fxns, data_loader=val_loader, split='val', history=history, model_dir=model_dir, weights=best_model_wts, amp=args.amp)

    # Evaluate on test set
    evaluate(model=model, tasks=tasks, loss_fxns=loss_fxns, data_loader=test_loader, split='test', history=history, model_dir=model_dir, weights=best_model_wts, amp=args.amp)

if __name__ == '__main__':
    init_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/gih5/echo_data')
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--model_name', type=str, default='3dresnet18', choices=['3dresnet18', 'frame_transformer'])
    parser.add_argument('--arch', type=str, default='', choices=['', 'resnet18', 'resnet50', 'convnext_tiny', 'convnext_small', 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'convnextv2_tiny.fcmae', 'transxnet_t', 'swin_v2_s', 'convnext_base.fb_in22k_ft_in1k_384', 'convnext_base.fb_in22k_ft_in1k'])
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--fc_dropout', type=float, default=0.)

    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pooling', type=str, default='mean', choices=['', 'mean', 'mean-max'])
    parser.add_argument('--transformer_dropout', type=float, default=0.)

    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--num_clips', type=int, default=4)
    parser.add_argument('--normalization', type=str, default='', choices=['', 'imagenet', 'kinetics'])
    parser.add_argument('--augment', action='store_true', default=False)

    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_class_weights', action='store_true', default=False)
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.)
    parser.add_argument('--cos_anneal', action='store_true', default=False)
    parser.add_argument('--T_0', type=int, default=5)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--reduce_lr', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    main(args)