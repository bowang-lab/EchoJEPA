import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import torch

from torchvision import tv_tensors
from torchvision.transforms import v2

class EchoNetPediatricDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_df, tasks, split, fold=None, clip_len=16, num_clips=4, augment=False, normalization=''):
        self.data_dir = data_dir
        self.data_df = data_df
        self.tasks = tasks
        self.split = split
        self.fold = fold
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.augment = augment
        self.normalization = normalization
        
        # Subset for specified cross-validation fold
        if fold is not None:
            self.data_df['split'] = self.data_df['fold'].apply(self._get_split)  # "Split" col = cross-validation fold ID
            self.data_df = self.data_df[self.data_df['split'] == split].reset_index(drop=True)

        # Set mean of each task (for current split)
        for task in self.tasks:
            if task.task_name == 'EF':
                task.mean = np.nanmean(self.data_df[task.task_name].values)
            else:
                task.mean = -1

        print(f'--- {split} (fold {fold}) ---')
        print(self.data_df)

        print(len(self.tasks))
        for task in self.tasks:
            print(vars(task))

        if self.normalization == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'kinetics':  # kinetics-400
            self.mean = np.array([0.43216, 0.394666, 0.37645])
            self.std = np.array([0.22803, 0.22145, 0.216989])
        elif self.normalization == 'echo-clip':
            self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        else:
            self.mean = None
            self.std = None

        if self.augment:
            trsf_list = [
                v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        else:
            trsf_list = [
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        
        if self.normalization != '':
            trsf_list += [v2.Normalize(mean=self.mean, std=self.std)]
        
        self.transform = v2.Compose(trsf_list)

    def _get_split(self, fold):
    # Following https://github.com/bryanhe/dynamic/blob/pediatric/scripts/cross_validate_pediatric.py
        if fold == self.fold:
            return "test"
        elif fold == (self.fold + 1) % 10:
            return "val"
        return "train"

    def _load_clip(self, fpath, clip_len=16):
        capture = cv2.VideoCapture(fpath)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count < clip_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, frame_count-clip_len+1, size=1)[0]

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)

        v = []
        for i in range(clip_len):
            if i < frame_count:
                ret, frame = capture.read()

                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

                v.append(frame)
            else:
                v.append(frame)  # "last image carried forward" padding

        v = np.stack(v, axis=0)  # f x h x w x 3
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

        return v

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx, :]  # info for a video

        fname = row['FileName']
        acc_num = row['acc_num']
        view = row['view']
        EF = row['EF']

        if self.split == 'train' or self.num_clips == 1:
            x = self._load_clip(os.path.join(self.data_dir, view.upper(), 'Videos', fname), self.clip_len)
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
        else:
            x = []
            for _ in range(self.num_clips):
                x_ = self._load_clip(os.path.join(self.data_dir, view.upper(), 'Videos', fname), self.clip_len)
                x_ = self.transform(x_)
                x_ = torch.permute(x_, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
                x.append(x_)
            x = torch.stack(x, dim=1)

        out_dict = {'x': x, 'acc_num': acc_num, 'fname': fname, 'view': view, 'EF': torch.FloatTensor([EF])}

        return out_dict

class EchoNetDynamicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_df, tasks, split, clip_len=16, num_clips=4, augment=False, normalization=''):
        self.data_dir = data_dir
        self.data_df = data_df
        self.tasks = tasks
        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.augment = augment
        self.normalization = normalization
        
        # Subset for specified split
        if split is not None:
            self.data_df = self.data_df[self.data_df['Split'] == split.upper()].reset_index(drop=True)

        # Set mean of each task (for current split)
        for task in self.tasks:
            if task.task_name in ['EF', 'LVESV', 'LVEDV']:
                task.mean = np.nanmean(self.data_df[task.task_name.replace('LV', '')].values)  # Rectify naming: LVESV -> ESV, etc.
            else:
                task.mean = -1

        print('---', split, '---')
        print(self.data_df)

        print(len(self.tasks))
        for task in self.tasks:
            print(vars(task))

        if self.normalization == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'kinetics':  # kinetics-400
            self.mean = np.array([0.43216, 0.394666, 0.37645])
            self.std = np.array([0.22803, 0.22145, 0.216989])
        elif self.normalization == 'echo-clip':
            self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        else:
            self.mean = None
            self.std = None

        if self.augment:
            trsf_list = [
                v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        else:
            trsf_list = [
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True)
            ]

        if self.normalization != '':
            trsf_list += [v2.Normalize(mean=self.mean, std=self.std)]
            
        self.transform = v2.Compose(trsf_list)

    def _load_clip(self, fpath, frame_count, clip_len=16):
        capture = cv2.VideoCapture(fpath)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count < clip_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, frame_count-clip_len+1, size=1)[0]

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)

        v = []
        for i in range(clip_len):
            if i < frame_count:
                ret, frame = capture.read()

                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

                v.append(frame)
            else:
                v.append(frame)  # "last image carried forward"

        v = np.stack(v, axis=0)  # f x h x w x 3
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

        return v

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx, :]  # info for a video

        fname = row['FileName']
        frame_count = row['NumberOfFrames']
        EF = row['EF']
        ESV = row['ESV']
        EDV = row['EDV']

        if self.split == 'train' or self.num_clips == 1:
            x = self._load_clip(os.path.join(self.data_dir, 'Videos', f'{fname}.avi'), frame_count, self.clip_len)
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
        else:
            x = []
            for _ in range(self.num_clips):
                x_ = self._load_clip(os.path.join(self.data_dir, 'Videos', f'{fname}.avi'), frame_count, self.clip_len)
                x_ = self.transform(x_)
                x_ = torch.permute(x_, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
                x.append(x_)
            x = torch.stack(x, dim=1)

        out_dict = {'x': x, 'fname': fname, 'EF': torch.FloatTensor([EF]), 'LVESV': torch.FloatTensor([ESV]), 'LVEDV': torch.FloatTensor([EDV])}

        return out_dict

class EchoNetLVHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_df, tasks, split, clip_len=16, num_clips=4, augment=False, normalization=''):
        self.data_dir = data_dir
        self.data_df = data_df
        self.tasks = tasks
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.num_clips = num_clips
        self.augment = augment
        self.normalization = normalization
        
        # Subset for specified split
        if split is not None:
            self.data_df = self.data_df[self.data_df['split'] == split].reset_index(drop=True)

        # Set mean of each task (for current split)
        for task in self.tasks:
            if task.task_name in ['IVSd', 'LVIDd', 'LVPWd', 'LVIDs']:
                task.mean = np.nanmean(self.data_df[task.task_name].values)
            else:
                task.mean = -1

        print('---', split, '---')
        print(self.data_df)

        print(len(self.tasks))
        for task in self.tasks:
            print(vars(task))

        if self.normalization == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'kinetics':  # kinetics-400
            self.mean = np.array([0.43216, 0.394666, 0.37645])
            self.std = np.array([0.22803, 0.22145, 0.216989])
        else:
            self.mean = None
            self.std = None

        if self.augment:
            trsf_list = [
                v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        else:
            trsf_list = [
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        
        if self.normalization != '':
            trsf_list += [v2.Normalize(mean=self.mean, std=self.std)]
            
        self.transform = v2.Compose(trsf_list)

    def _load_clip(self, fpath, frame_count, clip_len=16):
        capture = cv2.VideoCapture(fpath)

        if frame_count == -1:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count < clip_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, frame_count-clip_len+1, size=1)[0]

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)

        v = []
        for i in range(clip_len):
            if i < frame_count:
                ret, frame = capture.read()

                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                
                v.append(frame)
            else:
                v.append(frame)  # "last image carried forward"

        v = np.stack(v, axis=0)  # f x h x w x 3
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

        return v

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx, :]  # info for a video

        fname = row['HashedFileName']
        video_dir = row['video_dir']
        frame_count = row['frames']
        IVSd = row['IVSd']
        LVIDd = row['LVIDd']
        LVPWd = row['LVPWd']
        LVIDs = row['LVIDs']

        if self.split == 'train' or self.num_clips == 1:
            x = self._load_clip(os.path.join(self.data_dir, video_dir, f'{fname}.avi'), frame_count, self.clip_len)
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
        else:
            x = []
            for _ in range(self.num_clips):
                x_ = self._load_clip(os.path.join(self.data_dir, video_dir, f'{fname}.avi'), frame_count, self.clip_len)
                x_ = self.transform(x_)
                x_ = torch.permute(x_, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
                x.append(x_)
            x = torch.stack(x, dim=1)

        out_dict = {'x': x, 'fname': fname}
        
        for task in self.tasks:
            is_nan = np.isnan(row[task.task_name])

            # if NAN, replace with -1 (will be masked later)
            out_dict[task.task_name] = torch.LongTensor([row[task.task_name] if not is_nan else -1]) if task.task_type == 'multi-class_classification' else torch.FloatTensor([row[task.task_name] if not is_nan else -1])
            out_dict[task.task_name+'_mask'] = torch.BoolTensor([not is_nan])

        return out_dict

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_df, tasks, split, clip_len=16, sampling_rate=1, num_clips=4, augment=False, normalization='', train=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.tasks = tasks
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.num_clips = num_clips
        self.augment = augment
        self.normalization = normalization
        self.train = train
        
        # Set mean of each task (for current split)
        for task in self.tasks:
            task.mean = np.nanmean(self.data_df[task.task_name].values)


        print('---', split, '---')
        print(self.data_df)

        print(len(self.tasks))
        for task in self.tasks:
            print(vars(task))

        if self.normalization == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'kinetics':  # kinetics-400
            self.mean = np.array([0.43216, 0.394666, 0.37645])
            self.std = np.array([0.22803, 0.22145, 0.216989])
        else:
            self.mean = None
            self.std = None

        if self.augment:
            trsf_list = [
                v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        else:
            trsf_list = [
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True)
            ]

        if self.normalization != '':
            trsf_list += [v2.Normalize(mean=self.mean, std=self.std)]
        
        self.transform = v2.Compose(trsf_list)

    def _load_clip(self, fpath, clip_len=16):
        capture = cv2.VideoCapture(fpath)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count < clip_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, frame_count-clip_len+1, size=1)[0]

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)

        v = []
        for i in range(clip_len):
            if i < frame_count:
                ret, frame = capture.read()

                v.append(frame)
            else:
                v.append(frame)  # "last image carried forward"

        v = np.stack(v, axis=0)  # f x h x w x 3
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

        return v

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx, :]  # info for a video

        fname = row['fname']
        video_dir = row['video_dir']
        acc_num = row['acc_num']
        video_num = row['video_num']
        view = row['simple_view_pred']
        doppler = row['doppler']

        if self.train or self.num_clips == 1:
            x = self._load_clip(os.path.join(self.data_dir, video_dir, fname))
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
        else:
            x = []
            for _ in range(self.num_clips):
                x_ = self._load_clip(os.path.join(self.data_dir, video_dir, fname))
                x_ = self.transform(x_)
                x_ = torch.permute(x_, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
                x.append(x_)
            x = torch.stack(x, dim=1)

        out_dict = {'x': x, 'acc_num': acc_num, 'video_num': video_num, 'view': view, 'fname': fname, 'doppler': doppler, 'video_dir': video_dir}
        for task in self.tasks:
            is_nan = np.isnan(row[task.task_name])

            # if NAN, replace with -1 (will be masked later regardless)
            out_dict[task.task_name] = torch.LongTensor([row[task.task_name] if not is_nan else -1]) if task.task_type == 'multi-class_classification' else torch.FloatTensor([row[task.task_name] if not is_nan else -1])
            out_dict[task.task_name+'_mask'] = torch.BoolTensor([not is_nan])

        return out_dict
