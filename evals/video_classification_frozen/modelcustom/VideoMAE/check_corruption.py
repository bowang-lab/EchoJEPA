import boto3
import pandas as pd
import io
import decord
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor  # <--- Missing import added
from tqdm import tqdm
import sys

# ================= CONFIGURATION =================
CSV_S3_URI = "s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/mimic_annotations_s3.csv"
SAMPLE_SIZE = 500  # Number of files to check
MAX_WORKERS = 16   # Parallel threads for speed
# =================================================

def get_s3_client():
    return boto3.client('s3')

def parse_s3_uri(uri):
    # s3://bucket/key -> bucket, key
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

def check_file_validity(s3_path):
    """
    Downloads file to RAM and attempts to open with Decord.
    Returns: (is_valid, error_message)
    """
    s3 = get_s3_client()
    try:
        bucket, key = parse_s3_uri(s3_path)
        
        # 1. Download to RAM
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            video_bytes = obj['Body'].read()
            file_stream = io.BytesIO(video_bytes)
        except s3.exceptions.NoSuchKey:
            return False, "404 Not Found"
        except Exception as e:
            return False, f"Download Error: {str(e)}"

        # 2. Check Size
        if file_stream.getbuffer().nbytes < 1000: # Less than 1KB is suspicious
            return False, f"Suspiciously Small ({file_stream.getbuffer().nbytes} bytes)"

        # 3. Try Decord (Video Reader)
        try:
            vr = decord.VideoReader(file_stream, num_threads=1)
            # Access a frame to ensure the moov atom and data are readable
            _ = vr[0] 
            return True, "Valid"
        except Exception as e:
            return False, f"Decord Error (Corrupt): {str(e)}"

    except Exception as e:
        return False, f"Unexpected: {str(e)}"

def main():
    print(f"[1/4] Downloading Manifest from {CSV_S3_URI}...")
    manifest_df = pd.read_csv(CSV_S3_URI, header=None)
    
    # Handle CSV format: Assume column 0 is the path
    # If the CSV has "path, label", split it.
    all_paths = []
    for line in manifest_df[0]:
        # Clean line and grab first part (path)
        path = line.strip().split(" ")[0].split(",")[0]
        if path.startswith("s3://"):
            all_paths.append(path)
            
    total_files = len(all_paths)
    print(f"      Found {total_files:,} videos in manifest.")

    # Sampling
    sample_n = min(SAMPLE_SIZE, total_files)
    print(f"[2/4] Randomly sampling {sample_n} files...")
    sampled_paths = random.sample(all_paths, sample_n)

    print(f"[3/4] Checking validity (Threads: {MAX_WORKERS})...")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(check_file_validity, p): p for p in sampled_paths}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=sample_n):
            path = future_to_path[future]
            is_valid, reason = future.result()
            results.append({
                "path": path,
                "valid": is_valid,
                "reason": reason
            })

    # Analysis
    df = pd.DataFrame(results)
    valid_count = df['valid'].sum()
    corrupt_count = sample_n - valid_count
    corruption_rate = (corrupt_count / sample_n) * 100

    print("\n" + "="*40)
    print("             RESULTS SUMMARY")
    print("="*40)
    print(f"Total Checked:   {sample_n}")
    print(f"Valid Files:     {valid_count}")
    print(f"Corrupt Files:   {corrupt_count}")
    print(f"Corruption Rate: {corruption_rate:.2f}%")
    print("="*40)

    if corrupt_count > 0:
        print("\nTop 5 Failure Reasons:")
        print(df[~df['valid']]['reason'].value_counts().head())
        print("\nExample Corrupt Paths:")
        for p in df[~df['valid']]['path'].head(3):
            print(f" - {p}")

if __name__ == "__main__":
    main()