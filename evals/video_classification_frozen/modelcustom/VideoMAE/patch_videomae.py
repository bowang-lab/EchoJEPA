# Save this as patch_videomae.py inside ~/user-default-efs/VideoMAE/
import os

# 1. Create the S3-enabled Dataset Class
dataset_code = r'''
import os
import boto3
import io
import numpy as np
import torch
import decord
from torch.utils.data import DistributedSampler, DataLoader
from timm.data.loader import MultiEpochsDataLoader

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, frames_per_clip=16, frame_step=4, num_workers=4, pin_mem=True):
        self.samples = []
        if isinstance(data_paths, str): data_paths = [data_paths]
        
        for p in data_paths:
            if not os.path.exists(p) and not p.startswith("s3://"):
                print(f"WARN: Skipping {p}")
                continue
            # Handle S3 CSV download if needed, otherwise assume local
            if p.startswith("s3://"):
                # Simplified: assume user downloaded CSV to local path already
                continue 
            
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        self.samples.append((parts[0], int(parts[1]) if len(parts)>1 else 0))

        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.s3_client = None 

    def _get_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            vr = self.loadvideo_decord(path)
            buffer = self._sample_from_vr(vr)
            return buffer, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zero tensor on failure to prevent crash
            return torch.zeros((3, self.frames_per_clip, 224, 224)), label

    def loadvideo_decord(self, sample):
        if sample.startswith("s3://"):
            client = self._get_s3_client()
            bucket, key = sample.replace("s3://", "").split("/", 1)
            obj = client.get_object(Bucket=bucket, Key=key)
            f = io.BytesIO(obj['Body'].read())
            return decord.VideoReader(f, num_threads=1)
        else:
            return decord.VideoReader(sample, num_threads=1)

    def _sample_from_vr(self, vr):
        # Simple interval sampling matching VideoMAE logic
        duration = len(vr)
        required = self.frames_per_clip * self.frame_step
        
        if duration < required:
            # Loop video if too short
            indices = np.arange(0, required, self.frame_step) % duration
        else:
            # Random start
            max_start = duration - required
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            indices = np.arange(start, start + required, self.frame_step)

        images = vr.get_batch(indices).asnumpy() # (T, H, W, C)
        # Normalize/Transform (Basic version: float 0-1, CHW)
        images = torch.from_numpy(images).float().permute(3, 0, 1, 2) / 255.0
        
        # Resize to 224x224 (decord might read original size)
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        return images

def make_videodataset(data_paths, batch_size, frames_per_clip, frame_step, num_workers, pin_mem, rank, world_size, log_dir):
    dataset = VideoDataset(data_paths, frames_per_clip, frame_step)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True
    )
    return dataset, loader, sampler
'''

with open("s3_dataset.py", "w") as f:
    f.write(dataset_code)
print("[OK] Created s3_dataset.py")

# 2. Patch run_mae_pretraining.py to use it
with open("run_mae_pretraining.py", "r") as f:
    code = f.read()

if "from s3_dataset import make_videodataset" not in code:
    code = "from s3_dataset import make_videodataset\n" + code
    
    # Replace dataset build call
    old_call = "dataset_train = build_pretraining_dataset(args)"
    new_call = """
    # S3 PATCHED DATASET
    dataset_train, data_loader_train, sampler_train = make_videodataset(
        data_paths=args.data_path, 
        batch_size=args.batch_size, 
        frames_per_clip=args.num_frames, 
        frame_step=args.sampling_rate,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        rank=utils.get_rank(),
        world_size=utils.get_world_size(),
        log_dir=args.log_dir
    )
    # EXISTING LOGIC BYPASSED BELOW
    """
    code = code.replace(old_call, new_call)
    
    # Comment out the old DataLoader creation block to avoid errors
    code = code.replace("data_loader_train = torch.utils.data.DataLoader", "# data_loader_train = torch.utils.data.DataLoader")
    
    with open("run_mae_pretraining.py", "w") as f:
        f.write(code)
    print("[OK] Patched run_mae_pretraining.py")
else:
    print("[INFO] Already patched.")