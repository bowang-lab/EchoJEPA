import subprocess
import sys
import os
import boto3
import csv
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Progress Tracker ---
class ProgressTracker:
    def __init__(self, total, log_interval=50):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.log_interval = log_interval
        self.start_time = time.time()

    def update(self, success=True):
        self.completed += 1
        if not success:
            self.failed += 1
            
        if self.completed % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            rate = self.completed / (elapsed + 1e-9) # avoid div by zero
            print(f"Progress: {self.completed}/{self.total} "
                  f"({(self.completed/self.total)*100:.1f}%) "
                  f"- Rate: {rate:.1f} vids/sec - Failed: {self.failed}")

# --- Core Logic ---
def install_dependencies():
    subprocess.check_call("apt-get update -qq && apt-get install -y ffmpeg > /dev/null", shell=True)

def process_video(raw_uri):
    # --- CRITICAL FIX: Split off the label if it exists ---
    # Input: "s3://bucket/path/video.mp4 0.2598..."
    # Output: "s3://bucket/path/video.mp4"
    s3_uri = raw_uri.strip().split()[0]
    
    s3 = boto3.client('s3')
    TARGET_BUCKET = 'echodata25'
    TARGET_PREFIX = 'results/uhn-lvef-224'
    
    try:
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        filename = os.path.basename(key)
        
        local_input = f"/tmp/{filename}"
        local_output = f"/tmp/resized_{filename}"
        
        s3.download_file(bucket, key, local_input)
        
        # FFmpeg: 224x224, quiet mode
        cmd = [
            'ffmpeg', '-y', '-i', local_input, 
            '-vf', 'scale=224:224', '-c:a', 'copy', 
            '-loglevel', 'error', local_output
        ]
        subprocess.run(cmd, check=True)
        
        target_key = f"{TARGET_PREFIX}/{filename}"
        s3.upload_file(local_output, TARGET_BUCKET, target_key)
        
        # Cleanup
        if os.path.exists(local_input): os.remove(local_input)
        if os.path.exists(local_output): os.remove(local_output)
        return True
        
    except Exception as e:
        # Print the CLEANED uri in the error log to verify the fix
        print(f"ERROR on {s3_uri}: {e}")
        return False

if __name__ == "__main__":
    install_dependencies()
    
    input_dir = '/opt/ml/processing/input'
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not input_files:
        sys.exit(1)
        
    chunk_path = os.path.join(input_dir, input_files[0])
    print(f"Worker started on chunk: {chunk_path}")
    
    video_uris = []
    with open(chunk_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: 
                # row[0] might contain "s3://... 0.123". We pass it raw to process_video
                # which now handles the splitting.
                video_uris.append(row[0])
            
    total_videos = len(video_uris)
    tracker = ProgressTracker(total_videos)
    
    print(f"Starting threads for {total_videos} videos...")
    
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = {executor.submit(process_video, uri): uri for uri in video_uris}
        
        for future in as_completed(futures):
            success = future.result()
            tracker.update(success)
            
    print("Chunk processing complete.")
