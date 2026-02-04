"""
Batch apply depth attenuation to EchoNet-Dynamic dataset.

Creates 3 output directories with different attenuation levels:
- echonet-dynamic-da075 (attenuation_rate=0.75)
- echonet-dynamic-da150 (attenuation_rate=1.5)
- echonet-dynamic-da215 (attenuation_rate=2.15)

Usage:
    python batch_depth_attenuation.py                    # Process TEST split only (default)
    python batch_depth_attenuation.py --split train      # Process TRAIN split only
    python batch_depth_attenuation.py --split all        # Process all splits

Requirements:
    pip install usaugment av numpy albumentations tqdm pandas
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import pandas as pd

# Import the processing function from apply_depth_attenuation.py
# We'll inline the necessary parts to avoid import issues


def process_single_video(args):
    """Process a single video with depth attenuation."""
    input_path, output_path, attenuation_rate = args
    
    # Import here to avoid multiprocessing issues
    import av
    import numpy as np
    from fractions import Fraction
    import albumentations as A
    from usaugment.albumentations import DepthAttenuation
    
    try:
        # Open input video
        input_container = av.open(input_path)
        input_stream = input_container.streams.video[0]
        
        # Get video properties
        width = input_stream.width
        height = input_stream.height
        fps = input_stream.average_rate
        if fps is None:
            fps = Fraction(30, 1)
        
        # Read first frame (not used for mask since we use full image)
        input_container.seek(0)
        
        # No mask - apply to entire image
        scan_mask = np.ones((height, width), dtype=np.uint8)
        
        # Reset to beginning
        input_container.seek(0)
        
        # Initialize output container - MP4 with H.264
        output_container = av.open(output_path, mode='w')
        output_stream = output_container.add_stream('libx264', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
        output_stream.options = {
            'crf': '18',
            'preset': 'fast',  # Faster encoding for batch processing
        }
        
        # Initialize depth attenuation transform
        transform = A.Compose(
            [DepthAttenuation(
                attenuation_rate=attenuation_rate,
                max_attenuation=0.0,
                p=1.0,
            )],
            additional_targets={"scan_mask": "mask"}
        )
        
        # Process each frame
        frame_num = 0
        for frame in input_container.decode(video=0):
            frame_array = frame.to_ndarray(format='rgb24')
            frame_normalized = frame_array.astype(np.float32) / 255.0
            
            result = transform(image=frame_normalized, scan_mask=scan_mask)
            augmented = result["image"]
            
            augmented = np.clip(augmented * 255, 0, 255).astype(np.uint8)
            
            out_frame = av.VideoFrame.from_ndarray(augmented, format='rgb24')
            out_frame.pts = frame_num
            
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)
            
            frame_num += 1
        
        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        input_container.close()
        output_container.close()
        
        return (input_path, True, None)
    
    except Exception as e:
        return (input_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch apply depth attenuation to EchoNet-Dynamic dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="echonet/echonetdynamic-2/EchoNet-Dynamic/Videos",
        help="Input directory containing AVI files",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default="echonet/echonetdynamic-2/EchoNet-Dynamic/FileList.csv",
        help="Path to FileList.csv",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="echonet/echonetdynamic-2",
        help="Base output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to process: train, val, test, or all. Default: test",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default: 8",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Define attenuation levels
    attenuation_configs = [
        ("echonet-dynamic-da075", 0.75),
        ("echonet-dynamic-da150", 1.50),
        ("echonet-dynamic-da215", 2.15),
    ]
    
    # Read FileList.csv
    print(f"Reading file list from: {args.file_list}")
    df = pd.read_csv(args.file_list)
    
    # Show split distribution
    split_counts = df['Split'].value_counts()
    print(f"\nDataset split distribution:")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")
    print()
    
    # Filter by split
    if args.split.lower() != "all":
        split_upper = args.split.upper()
        df_filtered = df[df['Split'] == split_upper]
        print(f"Filtering to {split_upper} split: {len(df_filtered)} videos")
    else:
        df_filtered = df
        print(f"Processing ALL splits: {len(df_filtered)} videos")
    
    # Get list of video filenames
    input_dir = Path(args.input_dir)
    video_files = []
    for filename in df_filtered['FileName']:
        video_path = input_dir / f"{filename}.avi"
        if video_path.exists():
            video_files.append(video_path)
        else:
            print(f"Warning: Video not found: {video_path}")
    
    video_files = sorted(video_files)
    
    if args.limit:
        video_files = video_files[:args.limit]
    
    print(f"\nFound {len(video_files)} videos to process")
    print(f"Using {args.workers} workers")
    print(f"Output base directory: {args.output_base}")
    print()
    
    # Process each attenuation level
    for output_subdir, attenuation_rate in attenuation_configs:
        output_dir = Path(args.output_base) / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=" * 60)
        print(f"Processing: {output_subdir} (attenuation_rate={attenuation_rate})")
        print(f"Output directory: {output_dir}")
        print(f"=" * 60)
        
        # Prepare task list
        tasks = []
        for video_path in video_files:
            output_path = output_dir / (video_path.stem + ".mp4")
            
            # Skip if already processed
            if output_path.exists():
                continue
            
            tasks.append((str(video_path), str(output_path), attenuation_rate))
        
        if not tasks:
            print(f"All videos already processed for {output_subdir}, skipping...")
            print()
            continue
        
        print(f"Processing {len(tasks)} videos ({len(video_files) - len(tasks)} already done)")
        
        # Process in parallel
        successful = 0
        failed = 0
        failed_videos = []
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=output_subdir) as pbar:
                for future in as_completed(futures):
                    input_path, success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        failed_videos.append((input_path, error))
                    pbar.update(1)
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
        
        if failed_videos:
            print(f"\nFailed videos:")
            for path, error in failed_videos[:10]:  # Show first 10
                print(f"  {path}: {error}")
            if len(failed_videos) > 10:
                print(f"  ... and {len(failed_videos) - 10} more")
            
            # Save failed list to file
            failed_log = output_dir / "failed_videos.txt"
            with open(failed_log, "w") as f:
                for path, error in failed_videos:
                    f.write(f"{path}\t{error}\n")
            print(f"  Full list saved to: {failed_log}")
        
        print()
    
    print("=" * 60)
    print("All processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()