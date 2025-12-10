# Audio2Pose

Dance is considered one of the world’s earliest art forms, a universal language with which cultures have long communicated, told stories, and passed down traditions even before language was created. Because of its nature as a physical form of expression, dance is often challenging to explain verbally and even more difficult to translate into data. This project proposes to develop a machine learning model capable of generating 3D dance sequences conditioned on musical input. The system will take an audio track as input and output a temporally aligned, smooth 3D rendering of a dancer performing movements appropriate to the song’s rhythm and style.

## Getting Started

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### System Requirements

- Python 3.8+
- FFmpeg (for audio/video processing)
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

## Dataset Setup

The dataset consists of **two separate components**:
1. **Videos** (MP4 files) - Downloaded via script
2. **3D Keypoints** (PKL files) - Downloaded separately from AIST++ repository

### Step 1: Download LA Hip-Hop Videos

Download the video files (MP4) from the AIST++ database:

```bash
python scripts/download_aist.py --download_folder data/raw/aistpp_hiphop_la --num_processes 4
```

This downloads LA Hip-Hop videos (files starting with `gLH_`) into `data/raw/aistpp_hiphop_la/`.

**Note:** You must agree to the [AIST Dance Video Database Terms of Use](https://aistdancedb.ongaaccel.jp/terms_of_use/) before downloading.

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
   mkdir -p data/hiphop_la/3d_pkl
   
   # Extract downloaded archive to a temporary location first
   # (Adjust command based on archive format: .zip, .tar.gz, etc.)
   unzip aist_plusplus_3d_keypoints.zip -d /tmp/aistpp_pkl/
   # OR
   tar -xzf aist_plusplus_3d_keypoints.tar.gz -C /tmp/aistpp_pkl/
   
   # Filter and copy only LA Hip-Hop PKL files (starting with gLH_)
   cp /tmp/aistpp_pkl/gLH_*.pkl data/hiphop_la/3d_pkl/
   ```

4. **Verify PKL Files:**
   ```bash
   # Check that LA Hip-Hop PKL files are present
   ls data/hiphop_la/3d_pkl/*.pkl | wc -l  # Should show 141 files
   
   # Check structure of one PKL file
   python scripts/inspect_pkl.py data/hiphop_la/3d_pkl/gLH_sBM_cAll_d16_mLH0_ch01.pkl
   ```

**What's in the PKL files:**
- `keypoints3d`: 3D joint positions with shape `(T, J, 3)` where:
  - `T` = number of frames (~720 for 24s at 30fps)
  - `J` = number of joints (17)
  - `3` = (x, y, z) coordinates
- `keypoints3d_optim`: Optimized version of keypoints

**Important:** Only LA Hip-Hop PKL files (starting with `gLH_`) should be placed in `data/hiphop_la/3d_pkl/`. If you download the full AIST++ dataset with all genres, filter only the LA Hip-Hop files.

### Step 3: Preprocess Dataset

Run the preprocessing script to:
- Extract 3D poses from PKL files in `data/hiphop_la/3d_pkl/` → `.npy` format
- Extract audio from videos → WAV format (16kHz mono)
- Extract beat/tempo annotations from audio → JSON format

```bash
python scripts/preprocess.py \
  --pkl_dir data/hiphop_la/3d_pkl \
  --video_dir data/raw/aistpp_hiphop_la \
  --output_dir data/hiphop_la
```

**Arguments:**
- `--pkl_dir`: Directory containing LA Hip-Hop PKL files (`data/hiphop_la/3d_pkl`)
- `--video_dir`: Directory containing downloaded MP4 videos
- `--output_dir`: Output directory for processed data (default: `data/hiphop_la`)

**Optional flags:**
- `--skip_poses`: Skip pose extraction (if already done)
- `--skip_audio`: Skip audio extraction (if already done)
- `--skip_beats`: Skip beat/tempo extraction (if already done)
- `--overwrite`: Overwrite existing files

This creates:
- `data/hiphop_la/3d_pkl/` - LA Hip-Hop 3D PKL files (141 files)
- `data/hiphop_la/3d_npy/` - Extracted 3D pose arrays (.npy, shape: T×17×3)
- `data/hiphop_la/audio/` - Matching audio WAV files (16kHz mono, ~24s each)
- `data/hiphop_la/beats/` - Beat/tempo annotations (JSON format)

### Step 4: Verify Data

**Inspect a PKL file:**
```bash
python scripts/inspect_pkl.py data/hiphop_la/3d_pkl/gLH_sBM_cAll_d16_mLH0_ch01.pkl
```

**Check data statistics:**
```bash
# Count files in each directory
ls data/hiphop_la/3d_pkl/*.pkl | wc -l      # Should be 141
ls data/hiphop_la/3d_npy/*.npy | wc -l      # Should be 141
ls data/hiphop_la/audio/*.wav | wc -l       # Should be 141
ls data/hiphop_la/beats/*.json | wc -l      # Should be 141
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

## 3D Skeleton Visualization

### Overview

The project includes tools to render 3D skeleton visualizations overlaid on video frames with audio synchronization. This uses AIST++ 3D keypoints, camera parameters, and 2D video frames to create accurate skeleton overlays.

**Features:**
- **3D to 2D Projection**: Uses camera parameters to project 3D skeleton joints onto 2D video frames
- **AIST++ Style Rendering**: 
  - Spheres for joints
  - Lines/cylinders for bones
  - Color-coded body parts (torso: yellow, left arm: red, right arm: blue, left leg: green, right leg: magenta)
- **Smooth Motion**: Applies temporal smoothing for fluid animation
- **Audio Sync**: Combines video with audio for synchronized playback
- **Multi-Camera Support**: Supports all 9 camera views from AIST++

### Single Video Rendering

**Basic usage (auto-detection):**
```bash
python scripts/render_skeleton_video.py \
  data/raw/aistpp_hiphop_la/gLH_sBM_c01_d16_mLH0_ch01.mp4 \
  --output output/skeleton_video.mp4
```

The script will auto-detect:
- 3D pose file: `data/hiphop_la/3d_npy/gLH_sBM_cAll_d16_mLH0_ch01.npy`
- Audio file: `data/hiphop_la/audio/gLH_sBM_c01_d16_mLH0_ch01.wav` (or `cAll` version)
- Camera: Extracted from video filename (e.g., `c01`)

**Specify files explicitly:**
```bash
python scripts/render_skeleton_video.py \
  data/raw/aistpp_hiphop_la/gLH_sBM_c01_d16_mLH0_ch01.mp4 \
  --pose_3d data/hiphop_la/3d_npy/gLH_sBM_cAll_d16_mLH0_ch01.npy \
  --audio data/hiphop_la/audio/gLH_sBM_cAll_d16_mLH0_ch01.wav \
  --cameras data/hiphop_la/cameras/setting6.json \
  --output output/skeleton_video.mp4
```

**Customization options:**
```bash
python scripts/render_skeleton_video.py \
  data/raw/aistpp_hiphop_la/gLH_sBM_c01_d16_mLH0_ch01.mp4 \
  --output output/skeleton_video.mp4 \
  --camera c01 \
  --joint-radius 6 \
  --bone-thickness 4 \
  --smoothing-window 7 \
  --show-labels  # Show joint numbers
```

**Available options:**
- `--pose_3d PATH`: Path to 3D pose NPY file (shape: T, J, 3). Auto-detected if not specified.
- `--cameras PATH`: Path to camera parameters JSON (default: `data/hiphop_la/cameras/setting6.json`)
- `--audio PATH`: Path to audio file (WAV/MP3). Auto-detected if not specified.
- `--output PATH`: Output video path (required)
- `--camera NAME`: Camera name (e.g., `c01`). Auto-detected from video filename if not specified.
- `--no-smooth`: Disable pose smoothing
- `--smoothing-window INT`: Smoothing window size (default: 5)
- `--joint-radius INT`: Radius of joint spheres (default: 5)
- `--bone-thickness INT`: Thickness of bone lines (default: 3)
- `--fps FLOAT`: Output FPS (if not specified, uses video FPS)
- `--show-labels`: Show joint numbers on skeleton (for debugging)

### Batch Rendering

**Render multiple videos:**
```bash
python scripts/batch_render_skeleton.py \
  --video-dir data/raw/aistpp_hiphop_la \
  --output-dir output \
  --max-files 10
```

**Render one video per unique audio:**
```bash
python scripts/render_unique_audio_videos.py \
  --output-dir output/unique_audio
```

This script finds all unique audio files (by MD5 hash) and creates one skeleton video for each. Since multiple camera views share the same audio, this generates 12 unique videos (one per dance sequence).

```
Audio2Pose/
├── scripts/              # Utility scripts
│   ├── download_aist.py          # Download AIST++ videos
│   ├── preprocess.py              # Dataset preprocessing
│   ├── inspect_pkl.py             # Inspect PKL files
│   ├── render_skeleton_video.py   # Main skeleton rendering script
│   ├── batch_render_skeleton.py   # Batch rendering
│   ├── render_unique_audio_videos.py  # Render per unique audio
│   └── extract_2d.py               # Extract 2D keypoints
├── src/                  # Source code
│   ├── data/             # Data loading and processing
│   │   ├── loader.py              # Load pose/audio samples
│   │   ├── extract.py              # Extract poses/keypoints
│   │   └── inspect_pkl.py          # PKL inspection utilities
│   └── utils/            # Utilities
│       └── skeleton_viz.py         # Skeleton visualization core
├── data/                 # Dataset files
│   ├── raw/              # Raw downloaded data
│   └── hiphop_la/        # Processed LA Hip-Hop data
│       ├── 3d_pkl/       # 3D PKL files
│       ├── 3d_npy/       # 3D pose arrays (.npy)
│       ├── audio/        # Audio files (.wav)
│       ├── beats/        # Beat annotations (.json)
│       └── cameras/      # Camera parameters (.json)
├── output/               # Generated videos
└── requirements.txt      # Python dependencies
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `numpy>=1.24` - Numerical computing
- `opencv-python>=4.8` - Video processing and rendering
- `scipy>=1.10` - Scientific computing (rotation conversions)
- `soundfile>=0.12` - Audio I/O
- `librosa>=0.10.1` - Audio analysis
- `matplotlib>=3.7` - Plotting and visualization
- `torch>=2.2` - PyTorch (for future ML models)

## License

This project uses data from the AIST++ dataset. Please refer to the [AIST++ Dataset License](https://google.github.io/aistplusplus_dataset/license.html) for terms of use.
