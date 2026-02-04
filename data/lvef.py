#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

import os


# In[6]:


import pandas as pd

# Load everything (fast)
df_color = pd.read_parquet("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/color_inference_18m/predictions.parquet")
df_view = pd.read_parquet("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/view_inference_18m/predictions.parquet")


# In[3]:


df_color.head()


# In[4]:


df_view.head()


# In[41]:


len(df_view)


# In[49]:


# --- 1. Color Prediction Stats ---
print("--- Color Model Performance ---")
# Group by the class label ('is_color') and calculate mean/count for confidence
color_stats = df_color.groupby('is_color')['confidence'].agg(['mean', 'count', 'std'])
print(color_stats)

# --- 2. View Prediction Stats ---
print("\n--- View Model Performance ---")
# Group by the class label ('prediction') and calculate mean/count for confidence
view_stats = df_view.groupby('prediction')['confidence'].agg(['mean', 'count', 'std'])
print(view_stats)


# # Syngo

# In[5]:


data_dir = 'data/dataset/Lee_Echo_Syngo'
adler = pd.read_csv(os.path.join(data_dir, 'Adler.csv'))
analytics_report = pd.read_csv(os.path.join(data_dir, 'Analytics_Report.csv'))
analytics_study = pd.read_csv(os.path.join(data_dir, 'AnalyticsStudy.csv'))
department = pd.read_csv(os.path.join(data_dir, 'Department.csv'))
field_map = pd.read_csv(os.path.join(data_dir, 'FieldMap.csv'))
measurement_type = pd.read_csv(os.path.join(data_dir, 'MeasurementType.csv'))
modalities = pd.read_csv(os.path.join(data_dir, 'Modalities.csv'))
observations = pd.read_csv(os.path.join(data_dir, 'Observations.csv'))
study_details = pd.read_csv(os.path.join(data_dir, 'StudyDetails.csv'))


# In[6]:


aws_uhn = pd.read_csv('aws/aws_uhn.csv', index_col=0)
print(aws_uhn.shape)


# In[7]:


data_dir = 'data/dataset/Lee_Syngo_AnalyticMeasurement'
analytics_measure = pd.read_csv(os.path.join(data_dir, 'AnalyticsMeasure_Total.csv'))


# In[8]:


analytics_measure.head()


# In[9]:


len(analytics_measure)


# In[10]:


syn_lvef = analytics_measure.loc[analytics_measure['MeasurementName'] == 'LV EF, MOD BP']


# In[11]:


syn_lvef.head()


# In[12]:


syn_df = syn_lvef


# In[13]:


# Create a boolean mask to identify matching StudyRef values
matching_mask = syn_df['StudyRef'].isin(aws_uhn['STUDY_REF'])

# Count the number of matches (True values)
number_of_matches = matching_mask.sum()

# Get the total number of unique studies in mv_obs for context
total_unique_studies = syn_df['StudyRef'].nunique()

print(f"\nTotal unique StudyRef values in syn_df: {total_unique_studies}")
print(f"Number of 'StudyRef' values from syn_df that are in aws_uhn: {number_of_matches}")
print(f"Percentage of matching studies: {(number_of_matches / total_unique_studies) * 100:.2f}%")


# In[14]:


import pandas as pd

# Step 1: Merge the two DataFrames
# We'll merge 'syn_df' (lvef) with 'aws_uhn' using 'StudyRef' and 'STUDY_REF' as the keys.
merged_df = pd.merge(syn_df, aws_uhn, left_on='StudyRef', right_on='STUDY_REF', how='inner')

# Step 2: Define the desired columns in the correct order
final_columns = [
    'STUDY_REF',
    's3_key',
    'Value',
    'PATIENT_ID',
    'STUDY_DATE',
    'STUDY_TIME',
    'DeidentifiedStudyID',
    'OriginalStudyID'
]

# # Step 3: Create the new table with only the specified columns
syn_df_labels = merged_df[final_columns]

# convert numeric date to datetime
syn_df_labels['STUDY_DATE'] = pd.to_datetime(
    syn_df_labels['STUDY_DATE'].astype(str), 
    format='%Y%m%d'
)

# now sortable by date
syn_df_labels = syn_df_labels.sort_values('STUDY_DATE')
syn_df_labels['STUDY_DATE'] = syn_df_labels['STUDY_DATE'].dt.strftime('%Y-%m-%d')


# In[17]:


len(syn_df_labels)


# In[15]:


syn_df_labels.head()


# #### **To Do: Link to highest confidence + b/w A4C video in each study (maybe multiple, above a threshold). Do a patient-disjoint temporal split. Normalize regression values. Do the same for HeartLab.**

# In[26]:


import pandas as pd
import numpy as np

def create_patient_disjoint_temporal_split(df, 
                                           patient_col='PATIENT_ID', 
                                           time_col='STUDY_DATE', 
                                           ratios=[0.7, 0.1, 0.2]):
    """
    Splits dataframe strictly by patient, ordered by time, respecting data volume ratios.
    """
    # 1. Get unique patients with their temporal metadata and study volume
    patient_meta = df.groupby(patient_col).agg(
        last_study_date=(time_col, 'max'),
        study_count=('s3_key', 'count') # Or any non-null col
    ).reset_index()

    # 2. Sort patients by time (Temporal aspect)
    patient_meta = patient_meta.sort_values(by='last_study_date')

    # 3. Calculate cumulative counts to find split indices based on VOLUME (not just patient count)
    total_studies = len(df)
    patient_meta['cumulative_studies'] = patient_meta['study_count'].cumsum()

    # Define thresholds
    train_cutoff = total_studies * ratios[0]
    val_cutoff = total_studies * (ratios[0] + ratios[1])

    # 4. Assign split labels
    def get_split_label(cumulative_count):
        if cumulative_count <= train_cutoff:
            return 'train'
        elif cumulative_count <= val_cutoff:
            return 'val'
        else:
            return 'test'

    patient_meta['split'] = patient_meta['cumulative_studies'].apply(get_split_label)

    # 5. Merge split labels back to original dataframe
    df_split = df.merge(patient_meta[[patient_col, 'split']], on=patient_col, how='left')

    # Create the subsets
    train_df = df_split[df_split['split'] == 'train'].copy()
    val_df = df_split[df_split['split'] == 'val'].copy()
    test_df = df_split[df_split['split'] == 'test'].copy()

    return train_df, val_df, test_df

# --- Usage ---

# Ensure STUDY_DATE is datetime (looks like you did this in Image 3, but good to double check)
syn_df_labels['STUDY_DATE'] = pd.to_datetime(syn_df_labels['STUDY_DATE'])

train_df, val_df, test_df = create_patient_disjoint_temporal_split(syn_df_labels)

# --- Verification & Stats ---
print(f"Total Studies: {len(syn_df_labels)}")
print(f"Train: {len(train_df)} ({len(train_df)/len(syn_df_labels):.1%}) | Dates: {train_df['STUDY_DATE'].min().date()} to {train_df['STUDY_DATE'].max().date()}")
print(f"Val:   {len(val_df)} ({len(val_df)/len(syn_df_labels):.1%}) | Dates: {val_df['STUDY_DATE'].min().date()} to {val_df['STUDY_DATE'].max().date()}")
print(f"Test:  {len(test_df)} ({len(test_df)/len(syn_df_labels):.1%}) | Dates: {test_df['STUDY_DATE'].min().date()} to {test_df['STUDY_DATE'].max().date()}")

# Verify Patient Disjointness
train_pats = set(train_df['PATIENT_ID'])
val_pats = set(val_df['PATIENT_ID'])
test_pats = set(test_df['PATIENT_ID'])

assert len(train_pats.intersection(val_pats)) == 0, "Leakage between Train and Val!"
assert len(train_pats.intersection(test_pats)) == 0, "Leakage between Train and Test!"
assert len(val_pats.intersection(test_pats)) == 0,   "Leakage between Val and Test!"

print("\nSUCCESS: Splits are patient-disjoint and temporally ordered.")


# In[39]:


print(len(train_df))
train_df.head()


# # A4C

# In[50]:


import pandas as pd
import os

color_threshold = 0.85
view_threshold = 0.8

def add_video_paths_to_splits(train_df, val_df, test_df, df_color, df_view):
    # --- Step 1: Filter Video Metadata (Verified Working) ---
    valid_color = df_color[(df_color['is_color'] == 'No') & (df_color['confidence'] > color_threshold)]
    valid_view = df_view[(df_view['prediction'] == 'A4C') & (df_view['confidence'] > view_threshold)]
    valid_videos = pd.merge(valid_color, valid_view, on='s3_uri', suffixes=('_color', '_view'))

    # --- Step 2: Fix the Join Key ---
    def extract_study_key(uri):
        # 1. Remove the bucket prefix
        # Input: s3://echodata25/results/echo-study/STUDY_UID/SERIES_UID/video.mp4
        clean_path = uri.replace("s3://echodata25/results/", "")

        # 2. Split into parts: ['echo-study', 'STUDY_UID', 'SERIES_UID', 'video.mp4']
        parts = clean_path.split('/')

        # 3. Take only the first two parts and ADD THE TRAILING SLASH to match train_df
        # Output: echo-study/STUDY_UID/
        return f"{parts[0]}/{parts[1]}/"

    valid_videos['study_join_key'] = valid_videos['s3_uri'].apply(extract_study_key)

    # --- Step 3: Merge with Splits ---
    train_expanded = pd.merge(train_df, valid_videos, left_on='s3_key', right_on='study_join_key', how='inner')
    val_expanded = pd.merge(val_df, valid_videos, left_on='s3_key', right_on='study_join_key', how='inner')
    test_expanded = pd.merge(test_df, valid_videos, left_on='s3_key', right_on='study_join_key', how='inner')

    # Cleanup
    cols_to_drop = ['study_join_key']
    train_expanded.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    val_expanded.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test_expanded.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return train_expanded, val_expanded, test_expanded

# --- Run & Verify ---
train_final, val_final, test_final = add_video_paths_to_splits(
    train_df, val_df, test_df, df_color, df_view
)

print(f"Total -> {len(train_final) + len(val_df) + len(test_df)} videos")
print("")
print(f"Train: {len(train_df)} studies -> {len(train_final)} videos")
print(f"Val:   {len(val_df)} studies -> {len(val_final)} videos")
print(f"Test:  {len(test_df)} studies -> {len(test_final)} videos")


# In[51]:


train_final.head()


# In[53]:


# 1. Clean the data (LVEF must be 0-100)
# We use typical physiological bounds (e.g., 5% to 95%) or strictly 0-100
# depending on your clinical definition. 

def clean_lvef(df):
    initial_len = len(df)
    # Filter for physically valid ranges (0 to 100)
    df_clean = df[(df['Value'] >= 0) & (df['Value'] <= 100)].copy()
    print(f"Dropped {initial_len - len(df_clean)} rows with invalid values.")
    return df_clean


# In[55]:


# 1. Create the simplified DataFrames
cols_to_keep = ['s3_uri', 'Value']

train_simple = train_final[cols_to_keep].copy()
val_simple = val_final[cols_to_keep].copy()
test_simple = test_final[cols_to_keep].copy()

# Apply the basic cleaning (removing negatives/impossible values)
train_clean = clean_lvef(train_simple)
val_clean = clean_lvef(val_simple)
test_clean = clean_lvef(test_simple)

# 2. Function to print detailed statistics
def print_split_stats(name, df):
    print(f"================ {name} Set ================")
    print(f"Total Videos: {len(df)}")

    # Check for empty paths or null values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"WARNING: Found {missing_count} missing values!")

    # Calculate Skewness
    # > 0 means tail is on the right (more high values)
    # < 0 means tail is on the left (more low values / heart failure)
    skewness = df['Value'].skew()

    # Target Variable Statistics
    # We add 0.05 (5%) and 0.95 (95%) to the standard percentiles
    stats = df['Value'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    print("\nTarget ('Value') Statistics:")
    print(f"  Count: {int(stats['count'])}")
    print(f"  Mean:  {stats['mean']:.2f}")
    print(f"  Std:   {stats['std']:.2f}")
    print(f"  Skew:  {skewness:.2f}")
    print("-" * 20)
    print(f"  Min:   {stats['min']:.2f}")
    print(f"   5%:   {stats['5%']:.2f}  <-- Lower physiological bound (usually severe HF)")
    print(f"  25%:   {stats['25%']:.2f}")
    print(f"  Median:{stats['50%']:.2f}")
    print(f"  75%:   {stats['75%']:.2f}")
    print(f"  95%:   {stats['95%']:.2f}  <-- Upper physiological bound (hyperdynamic)")
    print(f"  Max:   {stats['max']:.2f}")
    print("\n")

# 3. Run the stats
print_split_stats("TRAIN", train_clean)
print_split_stats("VALIDATION", val_clean)
print_split_stats("TEST", test_clean)


# In[57]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Initialize the Scaler
# We use StandardScaler to center the data (Mean=0, Std=1)
scaler = StandardScaler()

# 2. Fit ONLY on the Training Set
# This prevents information leakage from the future/test set
train_values = train_clean['Value'].values.reshape(-1, 1)
scaler.fit(train_values)

print(f"Scaler fitted. Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")

# 3. Transform All Splits
# We create a new column 'norm_value' which the model will try to predict
train_clean['norm_value'] = scaler.transform(train_clean['Value'].values.reshape(-1, 1))
val_clean['norm_value']   = scaler.transform(val_clean['Value'].values.reshape(-1, 1))
test_clean['norm_value']  = scaler.transform(test_clean['Value'].values.reshape(-1, 1))

# 4. Save the Scaler
# CRITICAL: You need this file to convert predictions back to real LVEF % later
with open('lvef_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- Verification ---
print("\nNormalization Check (Train Set):")
print(train_clean['norm_value'].describe())
# Expected: Mean ~ 0.0, Std ~ 1.0

print("\nExample Data:")
print(train_clean[['Value', 'norm_value']].head(3))


# In[58]:


train_clean.head()


# In[66]:


import os
import pandas as pd

# 1. Setup Output Directory
output_dir = 'csv'
os.makedirs(output_dir, exist_ok=True)

def write_vjepa_csv(df, filename):
    """
    Writes DataFrame to V-JEPA 2 compatible CSV format:
    - Delimiter: Space
    - Columns: [s3_uri, norm_value]
    - No Header
    - No Index
    """
    output_path = os.path.join(output_dir, filename)

    # Select the relevant columns
    # Column 1: File Path (s3_uri)
    # Column 2: Label (norm_value for regression)
    export_df = df[['s3_uri', 'norm_value']].copy()

    # Write to CSV
    export_df.to_csv(
        output_path, 
        sep=' ',       # Space delimiter as required
        header=False,  # No header row
        index=False    # Do not write row numbers
    )

    print(f"✅ Saved {len(df)} rows to: {output_path}")

    # Verification: Print the first few lines to confirm format
    print(f"--- Preview ({filename}) ---")
    with open(output_path, 'r') as f:
        for _ in range(3):
            print(f.readline().strip())
    print("-" * 30 + "\n")

# 2. Write the files
# Assuming train_clean, val_clean, test_clean are your final dataframes
write_vjepa_csv(train_clean, 'a4c_b_lvef_train.csv')
write_vjepa_csv(val_clean, 'a4c_b_lvef_val.csv')
write_vjepa_csv(test_clean, 'a4c_b_lvef_test.csv')


# In[ ]:





# # Resize

# In[8]:


import pandas as pd
import numpy as np
import os
import sagemaker
import boto3

# --- Configuration ---
N_INSTANCES = 20           # Number of processing nodes
TEMP_BUCKET = 'echodata25' # Where to store the temporary chunk CSVs
TEMP_PREFIX = 'processing-metadata/input-chunks'
LOCAL_CSV_PATH = '/home/sagemaker-user/user-default-efs/vjepa2/data/csv'

# 1. Load Dataframes from local EFS
print("Loading local CSVs...")
df_train = pd.read_csv(os.path.join(LOCAL_CSV_PATH, 'a4c_b_lvef_train.csv'), header=None, names=['s3_uri', 'label'])
df_val   = pd.read_csv(os.path.join(LOCAL_CSV_PATH, 'a4c_b_lvef_val.csv'), header=None, names=['s3_uri', 'label'])
df_test  = pd.read_csv(os.path.join(LOCAL_CSV_PATH, 'a4c_b_lvef_test.csv'), header=None, names=['s3_uri', 'label'])

# 2. Combine and Shuffle (Shuffling ensures fair load balancing)
df_all = pd.concat([df_train, df_val, df_test]).sample(frac=1).reset_index(drop=True)
total_files = len(df_all)
print(f"Total videos to process: {total_files}")

# 3. Split into N chunks
chunks = np.array_split(df_all, N_INSTANCES)

# 4. Upload chunks to S3
# The processing instances cannot read your local EFS, so we must push these manifests to S3.
input_s3_uri = f's3://{TEMP_BUCKET}/{TEMP_PREFIX}'
os.makedirs('input_chunks', exist_ok=True)

print(f"Uploading {N_INSTANCES} chunks to {input_s3_uri}...")
for i, chunk in enumerate(chunks):
    chunk_filename = f'chunk_{i:03d}.csv'
    local_path = f'input_chunks/{chunk_filename}'

    # Save locally first
    chunk.to_csv(local_path, index=False, header=False, columns=['s3_uri'])

    # Upload to S3
    s3_path = f"{input_s3_uri}/{chunk_filename}"
    boto3.Session().resource('s3').Bucket(TEMP_BUCKET).upload_file(local_path, f"{TEMP_PREFIX}/{chunk_filename}")

print(f"✅ Data prepared. S3 Input Path: {input_s3_uri}")


# In[15]:


get_ipython().run_cell_magic('writefile', 'resize_script.py', 'import subprocess\nimport sys\nimport os\nimport boto3\nimport csv\nimport time\nfrom urllib.parse import urlparse\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\n\n# --- Progress Tracker ---\nclass ProgressTracker:\n    def __init__(self, total, log_interval=50):\n        self.total = total\n        self.completed = 0\n        self.failed = 0\n        self.log_interval = log_interval\n        self.start_time = time.time()\n\n    def update(self, success=True):\n        self.completed += 1\n        if not success:\n            self.failed += 1\n\n        if self.completed % self.log_interval == 0:\n            elapsed = time.time() - self.start_time\n            rate = self.completed / (elapsed + 1e-9) # avoid div by zero\n            print(f"Progress: {self.completed}/{self.total} "\n                  f"({(self.completed/self.total)*100:.1f}%) "\n                  f"- Rate: {rate:.1f} vids/sec - Failed: {self.failed}")\n\n# --- Core Logic ---\ndef install_dependencies():\n    subprocess.check_call("apt-get update -qq && apt-get install -y ffmpeg > /dev/null", shell=True)\n\ndef process_video(raw_uri):\n    # --- CRITICAL FIX: Split off the label if it exists ---\n    # Input: "s3://bucket/path/video.mp4 0.2598..."\n    # Output: "s3://bucket/path/video.mp4"\n    s3_uri = raw_uri.strip().split()[0]\n\n    s3 = boto3.client(\'s3\')\n    TARGET_BUCKET = \'echodata25\'\n    TARGET_PREFIX = \'results/uhn-lvef-224\'\n\n    try:\n        parsed = urlparse(s3_uri)\n        bucket = parsed.netloc\n        key = parsed.path.lstrip(\'/\')\n        filename = os.path.basename(key)\n\n        local_input = f"/tmp/{filename}"\n        local_output = f"/tmp/resized_{filename}"\n\n        s3.download_file(bucket, key, local_input)\n\n        # FFmpeg: 224x224, quiet mode\n        cmd = [\n            \'ffmpeg\', \'-y\', \'-i\', local_input, \n            \'-vf\', \'scale=224:224\', \'-c:a\', \'copy\', \n            \'-loglevel\', \'error\', local_output\n        ]\n        subprocess.run(cmd, check=True)\n\n        target_key = f"{TARGET_PREFIX}/{filename}"\n        s3.upload_file(local_output, TARGET_BUCKET, target_key)\n\n        # Cleanup\n        if os.path.exists(local_input): os.remove(local_input)\n        if os.path.exists(local_output): os.remove(local_output)\n        return True\n\n    except Exception as e:\n        # Print the CLEANED uri in the error log to verify the fix\n        print(f"ERROR on {s3_uri}: {e}")\n        return False\n\nif __name__ == "__main__":\n    install_dependencies()\n\n    input_dir = \'/opt/ml/processing/input\'\n    input_files = [f for f in os.listdir(input_dir) if f.endswith(\'.csv\')]\n\n    if not input_files:\n        sys.exit(1)\n\n    chunk_path = os.path.join(input_dir, input_files[0])\n    print(f"Worker started on chunk: {chunk_path}")\n\n    video_uris = []\n    with open(chunk_path, \'r\') as f:\n        reader = csv.reader(f)\n        for row in reader:\n            if row: \n                # row[0] might contain "s3://... 0.123". We pass it raw to process_video\n                # which now handles the splitting.\n                video_uris.append(row[0])\n\n    total_videos = len(video_uris)\n    tracker = ProgressTracker(total_videos)\n\n    print(f"Starting threads for {total_videos} videos...")\n\n    with ThreadPoolExecutor(max_workers=14) as executor:\n        futures = {executor.submit(process_video, uri): uri for uri in video_uris}\n\n        for future in as_completed(futures):\n            success = future.result()\n            tracker.update(success)\n\n    print("Chunk processing complete.")\n')


# In[17]:


import sagemaker
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput

current_region = boto3.Session().region_name
role = sagemaker.get_execution_role()
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")

sklearn_image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=current_region,
    version="1.2-1",
    image_scope="inference"
)

processor = ScriptProcessor(
    role=role,
    image_uri=sklearn_image_uri,
    command=['python3'],
    instance_count=N_INSTANCES,      
    instance_type='ml.c5.4xlarge',   
    volume_size_in_gb=50,
    base_job_name=f'resize-videos-fixed-{timestamp}' # Changed name to track easily
)

processor.run(
    code='resize_script.py',
    inputs=[
        ProcessingInput(
            # We can use the exact same input chunks from before
            source=input_s3_uri, 
            destination='/opt/ml/processing/input',
            s3_data_distribution_type='ShardedByS3Key'
        )
    ],
    wait=False
)

print(f"Job launched: {processor.latest_job.job_name}")


# In[18]:


import time
import boto3

# Configuration
TARGET_BUCKET = 'echodata25'
TARGET_PREFIX = 'results/uhn-lvef-224/'
EXPECTED_TOTAL = 256594 # Based on your preview (~176k + 26k + 53k)

s3 = boto3.client('s3')

print(f"Tracking progress for: s3://{TARGET_BUCKET}/{TARGET_PREFIX}")
print("Updates every 10 seconds...")

try:
    while True:
        # Count objects efficiently using Paginator
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=TARGET_BUCKET, Prefix=TARGET_PREFIX)

        count = 0
        for page in page_iterator:
            if 'Contents' in page:
                count += len(page['Contents'])

        percent = (count / EXPECTED_TOTAL) * 100
        print(f"\r✅ Processed: {count:,} / {EXPECTED_TOTAL:,} ({percent:.2f}%)", end="")

        if count >= EXPECTED_TOTAL:
            print("\n🎉 Job Complete!")
            break

        time.sleep(10)

except KeyboardInterrupt:
    print("\nStopped monitoring (Job is likely still running in background).")


# In[38]:


import boto3
import time

# Ensure this matches your running job name
job_name = processor.latest_job.job_name 

logs = boto3.client('logs')
log_group = '/aws/sagemaker/ProcessingJobs'

print(f"Checking logs for: {job_name}")

try:
    # 1. Find the log streams for this job (Default order is by Name)
    streams = logs.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=job_name
    )

    stream_list = streams.get('logStreams', [])

    if not stream_list:
        print("❌ No logs found yet. The instances are likely still provisioning.")
        print("   Wait 2-3 minutes. This is normal for the first 5-8 mins.")
    else:
        # Sort manually in Python to get the most recently active stream
        # (Some streams might be empty setup streams)
        stream_list.sort(key=lambda x: x.get('lastEventTimestamp', 0), reverse=True)

        target_stream = stream_list[0]['logStreamName']
        print(f"✅ Logs found! Reading stream: {target_stream}\n" + "-"*50)

        events = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=target_stream,
            startFromHead=False,
            limit=20
        )

        for e in events['events']:
            print(e['message'].strip())

except Exception as e:
    print(f"Error reading logs: {e}")


# In[39]:


import pandas as pd
import boto3
import os
import random
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
BASE_PATH = '/home/sagemaker-user/user-default-efs/vjepa2/data/csv'
NEW_S3_PREFIX = "s3://echodata25/results/uhn-lvef-224/"

# Define input and output mapping
files_to_process = [
    ("a4c_b_lvef_train.csv", "a4c_b_lvef_train_224px.csv"),
    ("a4c_b_lvef_val.csv",   "a4c_b_lvef_val_224px.csv"),
    ("a4c_b_lvef_test.csv",  "a4c_b_lvef_test_224px.csv")
]

# --- HELPER FUNCTIONS ---
def transform_uri(old_uri):
    """
    Takes deep S3 path: s3://.../folder/folder/video.mp4
    Returns flat new path: s3://echodata25/results/uhn-lvef-224/video.mp4
    """
    filename = os.path.basename(old_uri)
    return os.path.join(NEW_S3_PREFIX, filename)

def check_s3_existence(uri_list, sample_size=1000):
    """Checks if a random sample of URIs actually exists in S3"""
    s3 = boto3.client('s3')
    sample = random.sample(uri_list, min(len(uri_list), sample_size))

    print(f"\n🔍 Validating {len(sample)} random files in S3...")

    exists_count = 0
    missing_list = []

    for uri in tqdm(sample, desc="Checking S3"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        try:
            s3.head_object(Bucket=bucket, Key=key)
            exists_count += 1
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                missing_list.append(uri)
            else:
                print(f"Error checking {uri}: {e}")

    return exists_count, len(sample), missing_list

# --- MAIN EXECUTION ---
all_new_uris = []

print("🚀 Starting CSV Update...")

for input_name, output_name in files_to_process:
    input_full_path = os.path.join(BASE_PATH, input_name)
    output_full_path = os.path.join(BASE_PATH, output_name)

    # 1. Read (Space separated, no header)
    # Using engine='python' because generic separators can warn in C engine
    df = pd.read_csv(input_full_path, sep=' ', header=None, names=['s3_uri', 'label'], engine='python')

    # 2. Transform
    # We apply the transformation to the URI column
    df['s3_uri'] = df['s3_uri'].apply(transform_uri)

    # Collect for validation later
    all_new_uris.extend(df['s3_uri'].tolist())

    # 3. Save (Space separated, no header, no index)
    df.to_csv(output_full_path, sep=' ', header=False, index=False)

    print(f"✅ Created: {output_name} ({len(df)} rows)")
    # Print a quick preview of the new format
    print(f"   Sample: {df.iloc[0]['s3_uri']} {df.iloc[0]['label']}")

# --- VALIDATION STEP ---
print("-" * 50)
found, total, missing = check_s3_existence(all_new_uris, sample_size=1000)

print(f"\n📊 Validation Report:")
print(f"   Success Rate: {found}/{total} ({(found/total)*100:.1f}%)")

if missing:
    print(f"   ⚠️ WARNING: {len(missing)} files were missing. First 5 missing:")
    for m in missing[:5]:
        print(f"      - {m}")
else:
    print("   🎉 All checked files exist!")


# # Log

# In[3]:


get_ipython().system(' cat ../lvef_regression_0106_v1.log')


# # Visualization Examples

# In[7]:


import os
import boto3
from urllib.parse import urlparse

# 1. Select the top 3 highest confidence rows for each prediction category
top_examples = df_view.groupby('prediction', group_keys=False).apply(
    lambda x: x.nlargest(3, 'confidence')
)

# Display the selection to verify
print(f"Selected {len(top_examples)} files across {df_view['prediction'].nunique()} categories.")
display(top_examples[['prediction', 'confidence', 's3_uri']].head())

# 2. Setup download directory
output_dir = 'top_confidence_examples'
os.makedirs(output_dir, exist_ok=True)

# 3. Initialize S3 client
s3 = boto3.client('s3')

print(f"\nStarting download to folder: {output_dir}/ ...")

for index, row in top_examples.iterrows():
    uri = row['s3_uri']
    view = row['prediction']
    conf = row['confidence']

    # Parse the S3 URI
    parsed_uri = urlparse(uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')

    # Create a descriptive local filename
    # Format: View_Confidence_OriginalFilename
    original_name = os.path.basename(object_key)
    local_filename = f"{view}_{conf:.4f}_{original_name}"
    local_path = os.path.join(output_dir, local_filename)

    try:
        s3.download_file(bucket_name, object_key, local_path)
        print(f"Downloaded: {local_filename}")
    except Exception as e:
        print(f"FAILED to download {uri}: {e}")

print("\nDownload complete.")


# In[ ]:




