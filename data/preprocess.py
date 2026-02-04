import os
import cv2
import boto3
import pandas as pd
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Settings
TARGET_SIZE = (224, 224)
NUM_THREADS = 16

def get_s3_bucket_key(s3_uri):
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip('/')

def process_video(s3_uri, split_name, output_base_dir):
    try:
        bucket, key = get_s3_bucket_key(s3_uri)
        filename = key.split('/')[-1]

        save_dir = os.path.join(output_base_dir, split_name, 'a4c')
        os.makedirs(save_dir, exist_ok=True)

        local_input = f"/tmp/{filename}"
        local_output = os.path.join(save_dir, filename)

        if os.path.exists(local_output):
            return 1 # Skipped count

        s3 = boto3.client('s3')
        s3.download_file(bucket, key, local_input)

        cap = cv2.VideoCapture(local_input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(local_output, fourcc, 30.0, TARGET_SIZE)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, TARGET_SIZE)
            out.write(frame_resized)

        cap.release()
        out.release()

        if os.path.exists(local_input):
            os.remove(local_input)

        return 1 # Success count

    except Exception as e:
        # Print error but don't crash the worker
        print(f"Error processing {s3_uri}: {str(e)}")
        return 0 # Fail count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()

    # 1. Read CSV
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    if not input_files:
        print("No CSV found.")
        exit(1)

    chunk_file = os.path.join(args.input_dir, input_files[0])
    df = pd.read_csv(chunk_file, header=None, names=['s3_uri', 'split'])

    total_videos = len(df)
    print(f"Starting processing on {total_videos} videos with {NUM_THREADS} threads...")

    # 2. Parallel Execution with Clean Logging
    completed = 0
    start_time = time.time()
    log_interval = 500 # Log every 500 videos

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all jobs
        futures = [
            executor.submit(process_video, row.s3_uri, row.split, args.output_dir) 
            for row in df.itertuples()
        ]

        # Monitor as they finish
        for f in as_completed(futures):
            result = f.result()
            completed += result

            # Print status every 'log_interval'
            if completed % log_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                percent = (completed / total_videos) * 100
                remaining = (total_videos - completed) / rate if rate > 0 else 0

                print(f"Progress: {completed}/{total_videos} ({percent:.1f}%) | "
                      f"Rate: {rate:.1f} vid/s | ETA: {remaining/60:.1f} min")

    print(f"Chunk Complete. Total time: {(time.time() - start_time)/60:.2f} min")
