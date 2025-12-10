# Audio2Pose - Beat-Synchronized Pose Retrieval

Retrieve and visualize dance poses from audio using beat-synchronized retrieval from training data.

## Overview

This project retrieves pose sequences from a training dataset by matching audio beats and features. It extracts beats from input audio, finds similar segments in the training data, and creates synchronized dance sequences.

## Features

- **Beat-synchronized retrieval**: Matches audio beats to pose segments
- **BPM filtering**: Enhanced retrieval filters training data by tempo similarity
- **Skeleton visualization**: Renders pose sequences as animated skeleton videos
- **Multiple retrieval modes**: Basic and enhanced retrieval with improved matching

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Retrieve Poses from Audio

**Basic retrieval:**
```bash
python scripts/retrieve_poses.py \
  output/my_audio/your_audio.mp3 \
  --output output/poses.npy \
  --data-root data/hiphop_la \
  --pose-fps 30 \
  --segment-length 0.5
```

**Enhanced retrieval (with BPM filtering):**
```bash
python scripts/retrieve_poses_enhanced.py \
  output/my_audio/your_audio.mp3 \
  --output output/poses_enhanced.npy \
  --data-root data/hiphop_la \
  --pose-fps 30 \
  --segment-length 0.5 \
  --bpm-threshold 10.0
```

### 2. Render Skeleton Video

```bash
python scripts/render_skeleton.py \
  output/poses.npy \
  --output output/dance.mp4 \
  --audio output/my_audio/your_audio.mp3 \
  --motion-amplification 1.0 \
  --smoothing-window 10
```

## Scripts

### `retrieve_poses.py`
Basic beat-synchronized pose retrieval. Extracts beats from audio and retrieves matching pose segments from training data.

**Arguments:**
- `audio` - Input audio file (required)
- `--output` - Output pose .npy file (required)
- `--data-root` - Training data root directory (default: `data/hiphop_la`)
- `--pose-fps` - Pose frame rate (default: 30)
- `--segment-length` - Length of each pose segment in seconds (default: 1.0)
- `--hop-length` - Audio hop length (default: 512)
- `--blend-frames` - Number of frames to blend at transitions (default: 15)
- `--no-blend` - Disable transition blending

### `retrieve_poses_enhanced.py`
Enhanced retrieval with BPM filtering and improved audio matching. Filters training data by tempo similarity and uses better matching algorithms.

**Additional arguments:**
- `--bpm-threshold` - BPM difference threshold for filtering (default: 10.0)

### `render_skeleton.py`
Renders pose sequences as skeleton videos with audio synchronization.

**Arguments:**
- `pose_file` - Input pose .npy file (required)
- `--output` - Output video file (required)
- `--audio` - Audio file to sync with video (optional)
- `--motion-amplification` - Motion amplification factor (default: 1.0)
- `--smoothing-window` - Temporal smoothing window size (default: 5)

## How It Works

1. **Beat Detection**: Extracts beat times and tempo from input audio using librosa
2. **Audio Feature Extraction**: Computes MFCC, chroma, and spectral features
3. **Segment Matching**: For each beat, finds the best matching pose segment from training data based on:
   - Audio feature similarity
   - Beat alignment
   - Motion continuity
   - Diversity (avoids repeating segments)
4. **Pose Assembly**: Concatenates retrieved segments with smooth transitions
5. **Visualization**: Renders poses as skeleton videos

## Data Structure

Expected training data structure:
```
data/hiphop_la/
├── 3d_npy/          # 3D pose files (.npy)
├── audio/           # Audio files (.wav)
└── beats/           # Beat annotations (.json)
```

## Examples

**Fast-paced dance (0.5s segments):**
```bash
python scripts/retrieve_poses.py output/my_audio/song.mp3 \
  --output output/fast_poses.npy \
  --segment-length 0.5 \
  --no-blend

python scripts/render_skeleton.py output/fast_poses.npy \
  --output output/fast_dance.mp4 \
  --audio output/my_audio/song.mp3 \
  --smoothing-window 10
```

**Smooth dance (longer segments):**
```bash
python scripts/retrieve_poses.py output/my_audio/song.mp3 \
  --output output/smooth_poses.npy \
  --segment-length 1.2 \
  --blend-frames 20

python scripts/render_skeleton.py output/smooth_poses.npy \
  --output output/smooth_dance.mp4 \
  --audio output/my_audio/song.mp3 \
  --smoothing-window 15
```

## Tips

- **Segment length**: Shorter (0.3-0.5s) = more variety, longer (1.0-1.5s) = smoother
- **BPM matching**: Use `retrieve_poses_enhanced.py` for better tempo alignment
- **Smoothing**: Lower window (5-10) = sharper motion, higher (15-20) = smoother
- **Audio quality**: Clear beats work best (dance music, hip-hop, etc.)