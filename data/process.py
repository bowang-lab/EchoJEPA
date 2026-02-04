import os
import glob
import pydicom
import cv2
import numpy as np
from tqdm import tqdm
import sys

# --- Config ---
# SageMaker automatically mounts S3 inputs to this local path
INPUT_DIR = "/opt/ml/processing/input"
# SageMaker automatically uploads files from this local path to S3
OUTPUT_DIR = "/opt/ml/processing/output"

TARGET_RES = (112, 112)

def process_and_save(dcm_path):
    """Reads DICOM, resizes to 112x112, saves as MP4."""
    try:
        # 1. Read DICOM
        dcm = pydicom.dcmread(dcm_path)
        pixel_array = dcm.pixel_array
        
        # 2. Normalize Shape to (Frames, H, W, 3)
        # Handle Grayscale (H, W) -> (1, H, W, 3)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array]*3, axis=-1)
            pixel_array = np.expand_dims(pixel_array, axis=0)
            
        # Handle (Frames, H, W) or (H, W, C)
        elif len(pixel_array.shape) == 3:
            # If Multi-frame grayscale (Frames, H, W) -> Convert to RGB
            if dcm.NumberOfFrames > 1 and pixel_array.shape[0] == dcm.NumberOfFrames:
                pixel_array = np.stack([pixel_array]*3, axis=-1)
            # If Single Frame RGB (H, W, 3) -> Add frame dim
            elif pixel_array.shape[2] == 3:
                 pixel_array = np.expand_dims(pixel_array, axis=0)

        frames, h, w, c = pixel_array.shape
        
        # 3. Define Output Path
        # e.g. /opt/ml/processing/output/study_123.mp4
        filename = os.path.splitext(os.path.basename(dcm_path))[0] + ".mp4"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # 4. Write MP4
        # 'mp4v' is fast and compatible. Use 'avc1' if you need strict H.264
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, TARGET_RES)
        
        for i in range(frames):
            frame = pixel_array[i]
            # Resize
            frame_resized = cv2.resize(frame, TARGET_RES, interpolation=cv2.INTER_AREA)
            out.write(frame_resized.astype('uint8'))
        
        out.release()
        return True
        
    except Exception as e:
        # Log error but don't crash the job
        print(f"[ERROR] Failed {dcm_path}: {e}")
        return False

def main():
    # Find all .dcm files in the input directory (recursive)
    files = glob.glob(f"{INPUT_DIR}/**/*.dcm", recursive=True)
    print(f"Found {len(files)} DICOMs to process in this shard.")
    
    success_count = 0
    
    # Process files
    for dcm_file in tqdm(files, desc="Converting"):
        if process_and_save(dcm_file):
            success_count += 1

    print(f"Shard Complete. Processed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()
