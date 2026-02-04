"""
Apply depth attenuation augmentation to echocardiogram AVI videos.

Uses PyAV (FFmpeg) for video I/O and usaugment for the depth attenuation transform.

Usage:
    python apply_depth_attenuation.py input_video.avi output_video.avi

Requirements:
    pip install usaugment av numpy albumentations
"""

import argparse
import av
import numpy as np
from fractions import Fraction
from pathlib import Path

import albumentations as A
from usaugment.albumentations import DepthAttenuation


def create_scan_mask_from_frame(frame: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Create a scan mask by detecting non-black regions in the frame.
    
    This works well for echocardiograms where the scan region is brighter
    than the surrounding black background.
    
    Args:
        frame: Input frame (RGB or grayscale)
        threshold: Pixel intensity threshold to consider as part of scan region
        
    Returns:
        Binary mask (1 = scan region, 0 = background)
    """
    if frame.ndim == 3:
        # Convert to grayscale by averaging channels
        gray = np.mean(frame, axis=2).astype(np.uint8)
    else:
        gray = frame
    
    # Threshold to find non-black regions
    mask = (gray > threshold).astype(np.uint8)
    
    # Simple morphological cleanup using numpy operations
    # Dilation followed by erosion (closing)
    kernel_size = 5
    mask = _morphological_close(mask, kernel_size)
    # Erosion followed by dilation (opening)
    mask = _morphological_open(mask, kernel_size)
    
    return mask


def _morphological_dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simple dilation using max filter."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows.max(axis=(-1, -2))


def _morphological_erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simple erosion using min filter."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=1)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows.min(axis=(-1, -2))


def _morphological_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Closing: dilation followed by erosion."""
    return _morphological_erode(_morphological_dilate(mask, kernel_size), kernel_size)


def _morphological_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Opening: erosion followed by dilation."""
    return _morphological_dilate(_morphological_erode(mask, kernel_size), kernel_size)


def create_sector_mask(
    height: int,
    width: int,
    apex_y: float = 0.0,
    apex_x: float = 0.5,
    angle_degrees: float = 75,
    radius_fraction: float = 0.95,
) -> np.ndarray:
    """
    Create a geometric sector/fan-shaped mask typical of echocardiograms.
    
    Args:
        height: Image height
        width: Image width
        apex_y: Vertical position of sector apex (0.0 = top, 1.0 = bottom)
        apex_x: Horizontal position of sector apex (0.5 = center)
        angle_degrees: Half-angle of the sector in degrees
        radius_fraction: Radius of sector as fraction of image diagonal
        
    Returns:
        Binary mask (1 = scan region, 0 = background)
    """
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Apex position in pixels
    apex_px = int(apex_x * width)
    apex_py = int(apex_y * height)
    
    # Calculate distance and angle from apex
    dx = x - apex_px
    dy = y - apex_py
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dx, dy))  # Angle from vertical
    
    # Maximum radius
    max_radius = radius_fraction * np.sqrt(height**2 + width**2) / 2
    
    # Create sector mask
    within_radius = distance <= max_radius
    within_angle = np.abs(angle) <= angle_degrees
    
    mask = (within_radius & within_angle).astype(np.uint8)
    
    return mask


def apply_depth_attenuation_to_video(
    input_path: str,
    output_path: str,
    attenuation_rate: float = 2.0,
    max_attenuation: float = 0.0,
    mask_threshold: int = 10,
    use_geometric_mask: bool = False,
    sector_angle: float = 75,
    output_format: str = "mp4",
) -> None:
    """
    Apply depth attenuation to an echocardiogram video.
    
    Args:
        input_path: Path to input AVI video
        output_path: Path for output video
        attenuation_rate: Rate of intensity decay with depth (higher = stronger)
        max_attenuation: Minimum intensity at maximum depth (0.0 to 1.0)
        mask_threshold: Threshold for automatic mask detection
        use_geometric_mask: If True, use geometric sector mask instead of auto-detection
        sector_angle: Half-angle for geometric sector mask (degrees)
        output_format: Output format - 'mp4' (recommended), 'avi', or 'same'
    """
    # Open input video
    input_container = av.open(input_path)
    input_stream = input_container.streams.video[0]
    
    # Get video properties
    width = input_stream.width
    height = input_stream.height
    # Get fps as Fraction for PyAV compatibility
    fps = input_stream.average_rate
    if fps is None:
        fps = Fraction(30, 1)  # Default fallback
    frame_count = input_stream.frames or "unknown"
    
    print(f"Input video: {width}x{height}, {float(fps):.2f} fps, {frame_count} frames")
    
    # Read first frame to create mask
    input_container.seek(0)
    first_frame = None
    for frame in input_container.decode(video=0):
        first_frame = frame.to_ndarray(format='rgb24')
        break
    
    if first_frame is None:
        raise ValueError("Could not read first frame")
    
    # Create scan mask
    if use_geometric_mask:
        print(f"Using geometric sector mask (angle: {sector_angle}°)")
        scan_mask = create_sector_mask(height, width, angle_degrees=sector_angle)
    elif mask_threshold < 0:
        # No mask - apply to entire image
        print("No mask - applying to entire image")
        scan_mask = np.ones((height, width), dtype=np.uint8)
    else:
        print(f"Auto-detecting scan mask (threshold: {mask_threshold})")
        scan_mask = create_scan_mask_from_frame(first_frame, threshold=mask_threshold)
    
    # Reset to beginning
    input_container.seek(0)
    
    # Determine output format and adjust path if needed
    input_codec_name = input_stream.codec_context.name
    print(f"Input codec: {input_codec_name}")
    
    # Auto-adjust output path extension if format is specified
    if output_format == "mp4" and not output_path.endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        print(f"Output path adjusted to: {output_path}")
    
    # Initialize output container
    output_container = av.open(output_path, mode='w')
    
    # Choose codec based on output format
    if output_format == "mp4":
        # H.264 in MP4 container - most compatible
        output_stream = output_container.add_stream('libx264', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
        output_stream.options = {
            'crf': '18',  # Quality (0-51, lower is better)
            'preset': 'medium',
        }
    elif output_format == "avi" or output_format == "same":
        # MJPEG in AVI - good compatibility with medical imaging software
        output_stream = output_container.add_stream('mjpeg', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuvj420p'
        output_stream.options = {'qscale:v': '2'}  # High quality
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    # Initialize depth attenuation transform using albumentations Compose
    transform = A.Compose(
        [DepthAttenuation(
            attenuation_rate=attenuation_rate,
            max_attenuation=max_attenuation,
            p=1.0,
        )],
        additional_targets={"scan_mask": "mask"}
    )
    
    print(f"Applying depth attenuation (rate: {attenuation_rate}, max: {max_attenuation})")
    
    # Process each frame
    frame_num = 0
    for frame in input_container.decode(video=0):
        # Convert to numpy array (RGB)
        frame_array = frame.to_ndarray(format='rgb24')
        
        # Normalize to [0, 1] for the transform
        frame_normalized = frame_array.astype(np.float32) / 255.0
        
        # Apply depth attenuation using albumentations
        result = transform(image=frame_normalized, scan_mask=scan_mask)
        augmented = result["image"]
        
        # Convert back to uint8
        augmented = np.clip(augmented * 255, 0, 255).astype(np.uint8)
        
        # Create output frame
        out_frame = av.VideoFrame.from_ndarray(augmented, format='rgb24')
        out_frame.pts = frame_num
        
        # Encode and write
        for packet in output_stream.encode(out_frame):
            output_container.mux(packet)
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num} frames")
    
    # Flush encoder
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    input_container.close()
    output_container.close()
    
    print(f"Done! Processed {frame_num} frames. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply depth attenuation to echocardiogram videos"
    )
    parser.add_argument("input", type=str, help="Input video path")
    parser.add_argument("output", type=str, help="Output video path")
    parser.add_argument(
        "--attenuation-rate",
        type=float,
        default=2.0,
        help="Attenuation rate (higher = stronger darkening with depth). Default: 2.0",
    )
    parser.add_argument(
        "--max-attenuation",
        type=float,
        default=0.0,
        help="Minimum intensity at max depth (0.0-1.0). Default: 0.0",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=10,
        help="Threshold for auto mask detection (0-255). Default: 10",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Apply to entire image without masking (recommended for echocardiograms with black background)",
    )
    parser.add_argument(
        "--geometric-mask",
        action="store_true",
        help="Use geometric sector mask instead of auto-detection",
    )
    parser.add_argument(
        "--sector-angle",
        type=float,
        default=75,
        help="Half-angle for geometric sector mask (degrees). Default: 75",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="mp4",
        choices=["mp4", "avi", "same"],
        help="Output format: 'mp4' (recommended, most compatible), 'avi', or 'same'. Default: mp4",
    )
    
    args = parser.parse_args()
    
    # If --no-mask is set, use -1 as mask_threshold to indicate no masking
    mask_threshold = -1 if args.no_mask else args.mask_threshold
    
    apply_depth_attenuation_to_video(
        input_path=args.input,
        output_path=args.output,
        attenuation_rate=args.attenuation_rate,
        max_attenuation=args.max_attenuation,
        mask_threshold=mask_threshold,
        use_geometric_mask=args.geometric_mask,
        sector_angle=args.sector_angle,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()