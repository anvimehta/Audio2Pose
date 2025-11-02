# Audio2Pose

Machine Learning project scaffold. Initialize repo and push to GitHub.

## Getting Started
- Create a virtual environment and install your dependencies.
- Add project code under appropriate folders.

## AIST++ Dataset Setup

### Download LA Hip-Hop Videos
```bash
python scripts/download_aist.py --download_folder data/raw/aistpp_hiphop_la --num_processes 4
```

### Preprocess Dataset
Run the consolidated preprocessing script to filter PKLs, extract poses, and extract/copy audio:

```bash
python scripts/preprocess.py \
  --pkl_dir data/aistpp_3d \
  --video_dir data/raw/aistpp_hiphop_la \
  --output_dir data/hiphop_la
```

This creates:
- `data/hiphop_la/pkl/` - Filtered LA Hip-Hop PKL files
- `data/hiphop_la/poses_npy/` - Extracted 3D pose arrays (.npy)
- `data/hiphop_la/audio/` - Matching audio WAV files (16kHz mono)

### Inspect Data
```bash
python scripts/inspect_pkl.py data/hiphop_la/pkl/<file>.pkl
```

### Load Data Programmatically
```python
from src.data.loader import load_single_sample
pose, (wav, sr) = load_single_sample("<file_id>", data_root="data/hiphop_la")
```

## License
Add a license if applicable.
