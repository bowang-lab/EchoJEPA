"""
Batch apply Gaussian shadow augmentation to EchoNet-Dynamic dataset.

Creates 3 output directories with different shadow intensity levels:
- echonet-dynamic-gs-low  (strength=0.4, sigma=0.15) - Subtle shadow
- echonet-dynamic-gs-med  (strength=0.6, sigma=0.20) - Moderate shadow  
- echonet-dynamic-gs-high (strength=0.8, sigma=0.25) - Prominent shadow

The shadow position is randomized PER VIDEO but stays CONSTANT throughout
all frames of that video (realistic acoustic shadow behavior).

Usage:
    python batch_gaussian_shadow.py                      # Process TEST split only (default)
    python batch_gaussian_shadow.py --split train        # Process TRAIN split only
    python batch_gaussian_shadow.py --split all          # Process all splits
    python batch_gaussian_shadow.py --presets high       # Only process HIGH preset
    python batch_gaussian_shadow.py --presets low,med    # Process LOW and MED presets

Requirements:
    pip install av numpy tqdm pandas
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import pandas as pd


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================
# Each preset defines: (output_folder_suffix, strength, sigma_x, sigma_y)
#
# strength: Shadow darkness (0.0-1.0). Higher = darker center.
# sigma_x:  Horizontal spread as fraction of image width.
# sigma_y:  Vertical spread as fraction of image height.
#
# These values simulate realistic acoustic shadows from ribs/obstructions.
# ============================================================================

SHADOW_PRESETS = {
    "low": {
        "folder": "echonet-dynamic-gs-low",
        "strength": 0.4,
        "sigma_x": 0.15,
        "sigma_y": 0.15,
        "description": "Subtle shadow - mild obstruction",
    },
    "med": {
        "folder": "echonet-dynamic-gs-med",
        "strength": 0.6,
        "sigma_x": 0.20,
        "sigma_y": 0.20,
        "description": "Moderate shadow - typical rib shadow",
    },
    "high": {
        "folder": "echonet-dynamic-gs-high",
        "strength": 0.8,
        "sigma_x": 0.25,
        "sigma_y": 0.25,
        "description": "Prominent shadow - significant obstruction",
    },
}


def process_single_video(args):
    """Process a single video with constant Gaussian shadow."""
    input_path, output_path, strength, sigma_x, sigma_y, random_seed = args
    
    # Import here to avoid multiprocessing issues
    import av
    import numpy as np
    from fractions import Fraction
    
    try:
        # Set random seed for reproducibility (each video gets unique but reproducible shadow)
        np.random.seed(random_seed)
        
        # Open input video
        input_container = av.open(input_path)
        input_stream = input_container.streams.video[0]
        
        # Get video properties
        width = input_stream.width
        height = input_stream.height
        fps = input_stream.average_rate
        if fps is None:
            fps = Fraction(30, 1)
        
        # Reset to beginning
        input_container.seek(0)
        
        # ====================================================================
        # Generate STATIC shadow map (constant for entire video)
        # ====================================================================
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        
        # Random center position (biased towards center region for visibility)
        mu_x = int(np.random.uniform(0.2, 0.8) * width)
        mu_y = int(np.random.uniform(0.2, 0.8) * height)
        
        # Convert normalized sigmas to pixels
        sig_x_px = sigma_x * width
        sig_y_px = sigma_y * height
        
        # Gaussian shadow formula: shadow = 1 - strength * exp(...)
        shadow_map = 1.0 - strength * np.exp(
            -((xv - mu_x) ** 2 / (2 * sig_x_px**2) + (yv - mu_y) ** 2 / (2 * sig_y_px**2))
        )
        shadow_map = shadow_map.astype(np.float32)
        
        # Expand to 3 channels for RGB
        shadow_map_rgb = np.stack([shadow_map] * 3, axis=-1)
        
        # ====================================================================
        # Initialize output container - MP4 with H.264
        # ====================================================================
        output_container = av.open(output_path, mode='w')
        output_stream = output_container.add_stream('libx264', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
        output_stream.options = {
            'crf': '18',
            'preset': 'fast',  # Faster encoding for batch processing
        }
        
        # ====================================================================
        # Process each frame
        # ====================================================================
        frame_num = 0
        for frame in input_container.decode(video=0):
            # Convert to float [0, 1]
            frame_array = frame.to_ndarray(format='rgb24').astype(np.float32) / 255.0
            
            # Apply the CONSTANT shadow multiplier
            augmented = frame_array * shadow_map_rgb
            
            # Convert back to uint8
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
        
        return (input_path, True, None, mu_x, mu_y)
    
    except Exception as e:
        return (input_path, False, str(e), None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Batch apply Gaussian shadow to EchoNet-Dynamic dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset Configurations:
  low:  strength=0.4, sigma=0.15 - Subtle shadow (mild obstruction)
  med:  strength=0.6, sigma=0.20 - Moderate shadow (typical rib shadow)
  high: strength=0.8, sigma=0.25 - Prominent shadow (significant obstruction)

Examples:
  # Process test split with all presets:
  python batch_gaussian_shadow.py --split test

  # Process only the 'high' preset for training data:
  python batch_gaussian_shadow.py --split train --presets high

  # Process low and medium presets:
  python batch_gaussian_shadow.py --presets low,med

  # Quick test with 10 videos:
  python batch_gaussian_shadow.py --limit 10
        """
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
        "--presets",
        type=str,
        default="low,med,high",
        help="Comma-separated list of presets to process. Options: low, med, high. Default: low,med,high",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility. Default: 42",
    )
    
    args = parser.parse_args()
    
    # Parse presets
    requested_presets = [p.strip().lower() for p in args.presets.split(",")]
    for p in requested_presets:
        if p not in SHADOW_PRESETS:
            print(f"Error: Unknown preset '{p}'. Available: {list(SHADOW_PRESETS.keys())}")
            sys.exit(1)
    
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
    print(f"Presets to process: {requested_presets}")
    print(f"Base random seed: {args.seed}")
    print()
    
    # Show preset details
    print("Preset configurations:")
    for preset_name in requested_presets:
        cfg = SHADOW_PRESETS[preset_name]
        print(f"  {preset_name}: {cfg['description']}")
        print(f"         strength={cfg['strength']}, sigma=({cfg['sigma_x']}, {cfg['sigma_y']})")
    print()
    
    # Process each preset
    for preset_name in requested_presets:
        cfg = SHADOW_PRESETS[preset_name]
        output_dir = Path(args.output_base) / cfg["folder"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=" * 60)
        print(f"Processing: {cfg['folder']}")
        print(f"  {cfg['description']}")
        print(f"  strength={cfg['strength']}, sigma=({cfg['sigma_x']}, {cfg['sigma_y']})")
        print(f"  Output directory: {output_dir}")
        print(f"=" * 60)
        
        # Prepare task list
        tasks = []
        for idx, video_path in enumerate(video_files):
            output_path = output_dir / (video_path.stem + ".mp4")
            
            # Skip if already processed
            if output_path.exists():
                continue
            
            # Generate unique but reproducible seed per video
            # This ensures same video gets same shadow position across runs
            video_seed = args.seed + hash(video_path.stem) % (2**31)
            
            tasks.append((
                str(video_path),
                str(output_path),
                cfg["strength"],
                cfg["sigma_x"],
                cfg["sigma_y"],
                video_seed,
            ))
        
        if not tasks:
            print(f"All videos already processed for {cfg['folder']}, skipping...")
            print()
            continue
        
        print(f"Processing {len(tasks)} videos ({len(video_files) - len(tasks)} already done)")
        
        # Process in parallel
        successful = 0
        failed = 0
        failed_videos = []
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=preset_name.upper()) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    input_path, success, error, _, _ = result
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