import os
import cv2
import numpy as np

# --- Configuration ---
INPUT_DIR = './mimic_p10_mp4'       # Where your unmasked MP4s are
OUTPUT_DIR = './mimic_p10_masked'   # Where to save masked MP4s

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sector_mask(h, w):
    """
    Generates the 'Voxel Cone' sector mask as a 2D uint8 array.
    0 = Masked (Black), 255 = Visible (Keep).
    Logic adapted strictly from the reference script.
    """
    # Initialize mask as all WHITE (255) - i.e., keep everything initially
    mask = np.ones((h, w), dtype=np.uint8) * 255

    def draw_box(x, y, box_w, box_h):
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + box_w)), min(h, int(y + box_h))
        # Set region to BLACK (0) to mask it out
        mask[y1:y2, x1:x2] = 0

    # --- 1. Top & Bottom bars ---
    # Reference logic scales based on 480h / 640w
    draw_box(0, 0, w, h * (35/480))
    draw_box(0, h - h*(90/480), w, h * (120/480))

    # --- 2. Left-side stack ---
    draw_box(0, 0, w*(60/640), h)
    draw_box(0, 0, w*(90/640), h*0.55)
    draw_box(0, 0, w*(77/640), h*0.62)
    draw_box(0, 0, w*(130/640), h*0.3)
    draw_box(0, 0, w*(150/640), h*0.26)
    draw_box(0, 0, w*(220/640), h*0.20)
    draw_box(0, h*0.72, w*(105/640), h*0.3)

    # --- 3. Right-side stack ---
    draw_box(w - w*(220/640), 0, w*(220/640), h*0.20)
    draw_box(w - w*(145/640), 0, w*(120/640), h*0.49)
    draw_box(w - w*(130/640), 0, w*(120/640), h*0.51)
    draw_box(w - w*(115/640), 0, w*(120/640), h*0.53)
    draw_box(w - w*(90/640), 0, w*(120/640), h)
    draw_box(w - w*(105/640), h*0.68, w*(105/640), h*0.3)
    draw_box(w - w*(115/640), h*0.72, w*(105/640), h*0.3)

    return mask

def process_video_masking(file_path):
    filename = os.path.basename(file_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    # Skip if already exists
    if os.path.exists(save_path):
        print(f"Skipping {filename}: Already exists.")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error opening {filename}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. Generate the mask ONCE for this video resolution
    # (Efficiency gain: don't recalculate boxes every frame)
    mask = create_sector_mask(height, width)

    # Setup Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 2. Apply Mask
            # bitwise_and is faster than multiplication for uint8 images
            # It keeps pixels where mask is 255 (white) and blackens where mask is 0
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            out.write(masked_frame)
        
        print(f"Masked: {save_path} ({total_frames} frames)")
        
    except Exception as e:
        print(f"Failed to mask {filename}: {e}")
    finally:
        cap.release()
        out.release()

# --- Main Loop ---
print("Starting video masking...")
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]

if not files:
    print("No .mp4 files found.")
else:
    for f in files:
        process_video_masking(os.path.join(INPUT_DIR, f))
    print("Done!")