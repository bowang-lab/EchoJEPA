import glob
import cv2
import os
import numpy as np

def main():
    # Get all mp4 files in current directory
    files = glob.glob("*.mp4")
    
    if not files:
        print("No .mp4 files found in the current directory.")
        return

    print(f"Found {len(files)} files. Analyzing...")
    print("-" * 60)

    total_frames = 0
    total_duration = 0
    total_fps = 0
    resolutions = []
    
    valid_count = 0

    for filename in files:
        try:
            cap = cv2.VideoCapture(filename)
            
            if not cap.isOpened():
                print(f"Could not open {filename}")
                continue
                
            # Read metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Accumulate for averages
            total_frames += frame_count
            total_duration += duration
            total_fps += fps
            resolutions.append(f"{width}x{height}")
            
            valid_count += 1
            cap.release()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if valid_count == 0:
        print("Could not read stats from any files.")
        return

    # Calculate statistics
    avg_frames = total_frames / valid_count
    avg_duration = total_duration / valid_count
    avg_fps = total_fps / valid_count
    
    # Find most common resolution
    from collections import Counter
    common_res = Counter(resolutions).most_common(1)[0][0]

    # Print Results
    print(f"FILES PROCESSED:      {valid_count}")
    print(f"AVG FPS:              {avg_fps:.2f}")
    print(f"AVG FRAME COUNT:      {avg_frames:.2f}")
    print(f"AVG DURATION:         {avg_duration:.2f} seconds")
    print(f"COMMON RESOLUTION:    {common_res}")
    print("-" * 60)

if __name__ == "__main__":
    main()
