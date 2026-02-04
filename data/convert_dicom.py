import os
import pydicom
import numpy as np
import cv2

# --- Configuration ---
INPUT_DIR = './mimic_p10_dcm'      # Folder where you downloaded the S3 files
OUTPUT_DIR = './mimic_p10_mp4' # Folder to save videos
TARGET_SIZE = (336, 336)       # (Width, Height)
FPS = 30                       # Frames Per Second for output video

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_dicom(file_path):
    filename = os.path.basename(file_path)
    save_path = os.path.join(OUTPUT_DIR, filename.replace('.dcm', '.mp4'))
    
    try:
        ds = pydicom.dcmread(file_path)
        
        # Check if pixel data exists
        if 'PixelData' not in ds:
            print(f"Skipping {filename}: No pixel data found.")
            return

        # Extract pixel array (Frames, Height, Width)
        # Note: Some DICOMs are (Frames, H, W, Channels). This handles grayscale (H, W) or (F, H, W).
        pixel_array = ds.pixel_array
        
        # If the image is 2D (single frame), add a dimension to make it 3D (1, H, W)
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # --- Normalization ---
        # DICOM data can range widely (e.g., 0-4096). We must scale to 0-255 for MP4.
        pixel_array = pixel_array.astype(float)
        p_min = pixel_array.min()
        p_max = pixel_array.max()
        
        # Avoid division by zero
        if p_max - p_min != 0:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min)) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)
            
        pixel_array = pixel_array.astype(np.uint8)

        # --- Video Writing ---
        # OpenCV VideoWriter setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, FPS, TARGET_SIZE, isColor=True)

        for frame in pixel_array:
            # Resize frame to 336x336
            resized_frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # Convert Grayscale to RGB (MP4 usually expects 3 channels)
            # If your source is already RGB, you can skip the conversion, but most Echo DICOMs are monochrome or palette color.
            if resized_frame.ndim == 2:
                bgr_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
            else:
                # If already RGB (or YBR), assume BGR for OpenCV
                bgr_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

            out.write(bgr_frame)

        out.release()
        print(f"Success: {save_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# --- Main Loop ---
print("Starting conversion...")
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.dcm')]

if not files:
    print("No .dcm files found in the input directory.")
else:
    for f in files:
        process_dicom(os.path.join(INPUT_DIR, f))
    print("Done!")