# Audio2Pose

Dance is considered one of the world’s earliest art forms, a universal language with which cultures have long communicated, told stories, and passed down traditions even before language was created. Because of its nature as a physical form of expression, dance is often challenging to explain verbally and even more difficult to translate into data. This project proposes to develop a machine learning model capable of generating 3D dance sequences conditioned on musical input. The system will take an audio track as input and output a temporally aligned, smooth 3D rendering of a dancer performing movements appropriate to the song’s rhythm and style.

## Getting Started
- Create a virtual environment and install your dependencies.

## AIST++ Dataset Setup

The dataset consists of **two separate components**:
1. **Videos** (MP4 files) - Downloaded via script
2. **3D Keypoints** (PKL files) - Downloaded separately from AIST++ repository

### Step 1: Download LA Hip-Hop Videos

Download the video files (MP4) from the AIST++ database:

```bash
python scripts/download_aist.py --download_folder data/raw/aistpp_hiphop_la --num_processes 4
```

This downloads LA Hip-Hop videos (files starting with `gLH_`) into `data/raw/aistpp_hiphop_la/`.

**Note:** You must agree to the AIST Dance Video Database Terms of Use before downloading.

### Step 2: Download 3D Keypoints (PKL Files)

**Important:** The 3D pose data comes from **motion capture** (not from videos). These are stored in PKL files and are **separate from the videos** - they must be downloaded separately from the AIST++ dataset repository.

**Instructions:**

1. **Access AIST++ Dataset:**
   - Visit the [AIST++ Dataset website](https://google.github.io/aistplusplus_dataset/download.html)
   - You may need to register/agree to terms of use
   - Navigate to the "3D Keypoints" or "Motion Capture Data" section

2. **Download PKL Files:**
   - Download the 3D keypoints data package (typically contains all PKL files)
   - The PKL files contain motion capture data for all genres and sequences

3. **Extract and Organize:**
   ```bash
   # Create directory for LA Hip-Hop PKL files
   mkdir -p data/hiphop_la/pkl
   
   # Extract downloaded archive to a temporary location first
   # (Adjust command based on archive format: .zip, .tar.gz, etc.)
   unzip aist_plusplus_3d_keypoints.zip -d /tmp/aistpp_pkl/
   # OR
   tar -xzf aist_plusplus_3d_keypoints.tar.gz -C /tmp/aistpp_pkl/
   
   # Filter and copy only LA Hip-Hop PKL files (starting with gLH_)
   cp /tmp/aistpp_pkl/gLH_*.pkl data/hiphop_la/pkl/
   ```

4. **Verify PKL Files:**
   ```bash
   # Check that LA Hip-Hop PKL files are present
   ls data/hiphop_la/pkl/*.pkl | wc -l  # Should show 141 files
   
   # Check structure of one PKL file
   python scripts/inspect_pkl.py data/hiphop_la/pkl/gLH_sBM_cAll_d16_mLH0_ch01.pkl
   ```

**What's in the PKL files:**
- `keypoints3d`: 3D joint positions with shape `(T, J, 3)` where:
  - `T` = number of frames (~720 for 24s at 30fps)
  - `J` = number of joints (17)
  - `3` = (x, y, z) coordinates
- `keypoints3d_optim`: Optimized version of keypoints

**Important:** Only LA Hip-Hop PKL files (starting with `gLH_`) should be placed in `data/hiphop_la/pkl/`. If you download the full AIST++ dataset with all genres, filter only the LA Hip-Hop files.

### Step 3: Preprocess Dataset

Run the consolidated preprocessing script to:
- Extract 3D poses from PKL files in `data/hiphop_la/pkl/` → `.npy` format
- Extract audio from videos → WAV format (16kHz mono)
- Extract beat/tempo annotations from audio → JSON format

```bash
python scripts/preprocess.py \
  --pkl_dir data/hiphop_la/pkl \
  --video_dir data/raw/aistpp_hiphop_la \
  --output_dir data/hiphop_la
```

**Arguments:**
- `--pkl_dir`: Directory containing LA Hip-Hop PKL files (`data/hiphop_la/pkl`)
- `--video_dir`: Directory containing downloaded MP4 videos
- `--output_dir`: Output directory for processed data (default: `data/hiphop_la`)

**Note:** The `--pkl_dir` should point to where you placed the LA Hip-Hop PKL files in Step 2. If you have all PKL files in a different location and want the script to filter them, you can point `--pkl_dir` to that location and the script will filter and copy LA Hip-Hop files automatically.

**Optional flags:**
- `--skip_poses`: Skip pose extraction (if already done)
- `--skip_audio`: Skip audio extraction (if already done)
- `--skip_beats`: Skip beat/tempo extraction (if already done)
- `--overwrite`: Overwrite existing files

This creates (or updates):
- `data/hiphop_la/pkl/` - LA Hip-Hop PKL files (141 files, placed in Step 2)
- `data/hiphop_la/poses_npy/` - Extracted 3D pose arrays (.npy, shape: 720×17×3)
- `data/hiphop_la/audio/` - Matching audio WAV files (16kHz mono, ~24s each)
- `data/hiphop_la/beats/` - Beat/tempo annotations (JSON format with `tempo_bpm`, `beat_times`, `beat_frames`)

### Step 4: Verify Data

**Inspect a PKL file:**
```bash
python scripts/inspect_pkl.py data/hiphop_la/pkl/gLH_sBM_cAll_d16_mLH0_ch01.pkl
```

**Check data statistics:**
```bash
# Count files in each directory
ls data/hiphop_la/pkl/*.pkl | wc -l      # Should be 141
ls data/hiphop_la/poses_npy/*.npy | wc -l  # Should be 141
ls data/hiphop_la/audio/*.wav | wc -l    # Should be 141
ls data/hiphop_la/beats/*.json | wc -l   # Should be 141
```

### Load Data Programmatically

```python
from src.data.loader import load_single_sample

# Load pose and audio for a sample
file_id = "gLH_sBM_c01_d16_mLH0_ch01"  # Without extension
pose, (wav, sr) = load_single_sample(file_id, data_root="data/hiphop_la")

print(f"Pose shape: {pose.shape}")      # (720, 17, 3)
print(f"Audio shape: {wav.shape}")       # (384000,) for ~24s at 16kHz
print(f"Sample rate: {sr}")             # 16000
```

**Load beat annotations:**
```python
import json
from pathlib import Path

beats_file = Path("data/hiphop_la/beats/gLH_sBM_c01_d16_mLH0_ch01.json")
with open(beats_file) as f:
    beats_data = json.load(f)
    
print(f"Tempo: {beats_data['tempo_bpm']} BPM")
print(f"Beat times: {beats_data['beat_times'][:5]}")  # First 5 beats in seconds
```