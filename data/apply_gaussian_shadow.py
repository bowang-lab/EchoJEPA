"""
Apply CONSTANT Gaussian Shadow augmentation to echocardiogram AVI videos.

Unlike the standard usaugment implementation, this script generates ONE shadow 
position per video and applies it to every frame, preventing the shadow 
from "jumping" around.

Usage:
    python apply_gaussian_shadow.py input.avi output.avi --output-format avi --strength 0.6
    
    # No mask (recommended for echocardiograms - applies to entire frame)
    python apply_gaussian_shadow.py input.avi output.avi --no-mask --strength 0.6
    
    # Force shadow to the center of the screen
    python apply_gaussian_shadow.py input.avi output.avi --center-x 0.5 --center-y 0.5

Requirements:
    pip install av numpy
"""

import argparse
import av
import numpy as np
from fractions import Fraction


def create_scan_mask_from_frame(frame: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Create a scan mask by detecting non-black regions in the frame."""
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2).astype(np.uint8)
    else:
        gray = frame
    
    mask = (gray > threshold).astype(np.uint8)
    
    # Simple morphological cleanup (Close then Open)
    from numpy.lib.stride_tricks import sliding_window_view
    
    def morph_op(m, k, op):
        pad = k // 2
        padded = np.pad(m, pad, mode='constant', constant_values=(0 if op == 'max' else 1))
        windows = sliding_window_view(padded, (k, k))
        return windows.max(axis=(-1, -2)) if op == 'max' else windows.min(axis=(-1, -2))

    k = 5
    # Close (Dilate -> Erode)
    mask = morph_op(morph_op(mask, k, 'max'), k, 'min')
    # Open (Erode -> Dilate)
    mask = morph_op(morph_op(mask, k, 'min'), k, 'max')
    
    return mask


def create_sector_mask(height, width, sector_angle=75):
    """Create a geometric sector/fan-shaped mask."""
    y, x = np.ogrid[:height, :width]
    # Assume apex is top-center
    apex_px, apex_py = int(0.5 * width), 0
    
    dx = x - apex_px
    dy = y - apex_py
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dx, dy))
    
    max_radius = 0.95 * np.sqrt(height**2 + width**2) / 2
    mask = ((distance <= max_radius) & (np.abs(angle) <= sector_angle)).astype(np.uint8)
    return mask


def generate_static_shadow_map(height, width, strength, sigma_x, sigma_y, center_x=None, center_y=None):
    """
    Generate a single 2D shadow map using Gaussian math.
    Returns a float32 map where 1.0 = no shadow, <1.0 = shadowed.
    """
    x = np.arange(0, width)
    y = np.arange(0, height)
    xv, yv = np.meshgrid(x, y)

    # If center not provided, pick random location (ONCE)
    # But bias towards the center region for visibility
    if center_x is None:
        # Random position biased towards center (between 0.2 and 0.8)
        mu_x = int(np.random.uniform(0.2, 0.8) * width)
    else:
        mu_x = int(center_x * width)
        
    if center_y is None:
        # Random position biased towards center (between 0.2 and 0.8)
        mu_y = int(np.random.uniform(0.2, 0.8) * height)
    else:
        mu_y = int(center_y * height)

    # Convert normalized sigmas to pixels
    sig_x_px = sigma_x * width
    sig_y_px = sigma_y * height

    # The Gaussian Shadow Formula
    # shadow = 1 - strength * exp(...)
    shadow_map = 1.0 - strength * np.exp(
        -((xv - mu_x) ** 2 / (2 * sig_x_px**2) + (yv - mu_y) ** 2 / (2 * sig_y_px**2))
    )
    
    print(f"  Shadow center: ({mu_x}, {mu_y}) pixels")
    print(f"  Shadow sigma: ({sig_x_px:.1f}, {sig_y_px:.1f}) pixels")
    print(f"  Shadow map range: [{shadow_map.min():.3f}, {shadow_map.max():.3f}]")
    
    return shadow_map.astype(np.float32)


def apply_constant_shadow_to_video(
    input_path: str,
    output_path: str,
    strength: float = 0.5,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    center_x: float = None,
    center_y: float = None,
    mask_threshold: int = 10,
    use_geometric_mask: bool = False,
    no_mask: bool = False,
    sector_angle: float = 75,
    output_format: str = "mp4",
):
    """Apply constant Gaussian shadow to a video."""
    
    # 1. Open Input
    input_container = av.open(input_path)
    input_stream = input_container.streams.video[0]
    width = input_stream.width
    height = input_stream.height
    fps = input_stream.average_rate or Fraction(30, 1)
    total_frames = input_stream.frames or "unknown"
    
    # Get input codec info for reference
    input_codec = input_stream.codec_context.name
    input_pix_fmt = input_stream.codec_context.pix_fmt
    
    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {float(fps):.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Codec: {input_codec}, Pixel format: {input_pix_fmt}")

    # 2. Prepare Mask (Identify Ultrasound Region)
    input_container.seek(0)
    first_frame = next(input_container.decode(video=0)).to_ndarray(format='rgb24')
    input_container.seek(0)  # Reset

    if no_mask:
        print("  Mask: DISABLED (applying to entire frame)")
        scan_mask = np.ones((height, width), dtype=np.uint8)
    elif use_geometric_mask:
        print(f"  Mask: Geometric sector (angle={sector_angle}°)")
        scan_mask = create_sector_mask(height, width, sector_angle)
    else:
        print(f"  Mask: Auto-detect (threshold={mask_threshold})")
        scan_mask = create_scan_mask_from_frame(first_frame, threshold=mask_threshold)
    
    # Debug: Print mask statistics
    mask_pixels = np.sum(scan_mask)
    total_pixels = height * width
    mask_pct = 100.0 * mask_pixels / total_pixels
    print(f"  Mask coverage: {mask_pixels}/{total_pixels} pixels ({mask_pct:.1f}%)")
    
    if mask_pixels == 0:
        print("  WARNING: Mask is empty! Shadow will not be visible.")
        print("  Try using --no-mask or lowering --mask-threshold")

    # 3. Generate the STATIC Shadow Map (The "Constant" Part)
    print(f"\nGenerating shadow (strength={strength}, sigma={sigma_x}x{sigma_y}):")
    raw_shadow_map = generate_static_shadow_map(
        height, width, strength, sigma_x, sigma_y, center_x, center_y
    )

    # 4. Combine Shadow with Mask
    # Where mask is 1, apply shadow. Where mask is 0, keep original (multiply by 1.0)
    final_multiplier = np.where(scan_mask == 1, raw_shadow_map, 1.0).astype(np.float32)
    
    # Debug: Check final multiplier
    affected_pixels = np.sum(final_multiplier < 0.99)
    print(f"  Pixels affected by shadow: {affected_pixels} ({100.0 * affected_pixels / total_pixels:.1f}%)")
    print(f"  Final multiplier range: [{final_multiplier.min():.3f}, {final_multiplier.max():.3f}]")
    
    # Expand to 3 channels for RGB multiplication
    final_multiplier_rgb = np.stack([final_multiplier] * 3, axis=-1)

    # 5. Output Setup
    if output_format == "mp4" and not output_path.endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    elif output_format == "avi" and not output_path.endswith('.avi'):
        output_path = output_path.rsplit('.', 1)[0] + '.avi'
    
    print(f"\nOutput: {output_path}")
    print(f"  Format: {output_format.upper()}")
    
    output_container = av.open(output_path, mode='w')
    
    if output_format == "mp4":
        # H.264 with high quality settings
        out_stream = output_container.add_stream('libx264', rate=fps)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {
            'crf': '17',      # Lower = better quality (17-18 is visually lossless)
            'preset': 'slow', # Slower = better compression efficiency
        }
        print(f"  Codec: H.264 (CRF=17, preset=slow)")
    else:  # avi
        # Use high-quality MJPEG with forced best quality
        out_stream = output_container.add_stream('mjpeg', rate=fps)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = 'yuvj420p'
        # Force highest quality: qmin=1, qmax=1 means constant Q=1 (best)
        out_stream.options = {
            'qmin': '1',
            'qmax': '1',
        }
        print(f"  Codec: MJPEG (qmin=1, qmax=1 - highest quality)")

    # 6. Process Frames
    print(f"\nProcessing frames...")
    frame_num = 0
    for frame in input_container.decode(video=0):
        # Convert to float [0, 1]
        img = frame.to_ndarray(format='rgb24').astype(np.float32) / 255.0
        
        # Apply the CONSTANT shadow multiplier
        img = img * final_multiplier_rgb
        
        # Convert back to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Write frame
        out_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        out_frame.pts = frame_num
        
        for packet in out_stream.encode(out_frame):
            output_container.mux(packet)
            
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num} frames...")

    # Flush encoder
    for packet in out_stream.encode():
        output_container.mux(packet)
        
    input_container.close()
    output_container.close()
    
    print(f"\nDone! Processed {frame_num} frames.")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply constant Gaussian shadow to echocardiogram videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with recommended settings for echocardiograms:
  python apply_gaussian_shadow.py input.avi output.avi --no-mask --output-format avi

  # Strong, visible shadow in the center:
  python apply_gaussian_shadow.py input.avi output.avi --no-mask --strength 0.7 --center-x 0.5 --center-y 0.5

  # Random shadow position with custom size:
  python apply_gaussian_shadow.py input.avi output.avi --no-mask --sigma-x 0.15 --sigma-y 0.25
        """
    )
    
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    
    parser.add_argument(
        "--strength", 
        type=float, 
        default=0.5,
        help="Shadow darkness (0.0-1.0). Higher = darker. Default: 0.5"
    )
    parser.add_argument(
        "--sigma-x", 
        type=float, 
        default=0.2,
        help="Shadow width as fraction of image width. Default: 0.2"
    )
    parser.add_argument(
        "--sigma-y", 
        type=float, 
        default=0.2,
        help="Shadow height as fraction of image height. Default: 0.2"
    )
    parser.add_argument(
        "--center-x", 
        type=float, 
        default=None,
        help="Fixed X center (0.0-1.0). Default: Random (biased to center)"
    )
    parser.add_argument(
        "--center-y", 
        type=float, 
        default=None,
        help="Fixed Y center (0.0-1.0). Default: Random (biased to center)"
    )
    parser.add_argument(
        "--mask-threshold", 
        type=int, 
        default=10,
        help="Threshold for auto mask detection (0-255). Default: 10"
    )
    parser.add_argument(
        "--geometric-mask", 
        action="store_true",
        help="Use geometric sector mask instead of auto-detection"
    )
    parser.add_argument(
        "--no-mask", 
        action="store_true",
        help="Apply shadow to entire frame (RECOMMENDED for echocardiograms)"
    )
    parser.add_argument(
        "--output-format", 
        type=str, 
        default="mp4", 
        choices=["mp4", "avi"],
        help="Output format: 'mp4' or 'avi'. Default: mp4"
    )
    
    args = parser.parse_args()
    
    apply_constant_shadow_to_video(
        input_path=args.input,
        output_path=args.output,
        strength=args.strength,
        sigma_x=args.sigma_x,
        sigma_y=args.sigma_y,
        center_x=args.center_x,
        center_y=args.center_y,
        mask_threshold=args.mask_threshold,
        use_geometric_mask=args.geometric_mask,
        no_mask=args.no_mask,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()