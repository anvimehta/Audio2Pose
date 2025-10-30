# Audio2Pose

Machine Learning project scaffold. Initialize repo and push to GitHub.

## Getting Started
- Create a virtual environment and install your dependencies.
- Add project code under appropriate folders.

## AIST++ Dataset Setup
1. Download AIST++ from the official sources (`aistplusplus-api` and the 3D keypoints/SMPL assets). Place audio and 3D keypoints locally.
2. Organize files under this repository:

```
data/
  raw/
    aistpp_3d/
```

- If your files live elsewhere, pass the root to loaders via `data_root`.
- To inspect a 3D keypoints pickle and visualize the first frame:

```bash
python scripts/inspect_pkl.py data/aistpp_3d/<your_file>.pkl
```

- To load a single sample programmatically:

```python
from src.data.loader import load_single_sample
pose, (wav, sr) = load_single_sample("<file_id>", data_root="data/raw/aistpp_hiphop")
```

### Extract 3D poses from PKL to NPY
- Extract all PKLs in a folder to NPYs (skips existing files):

```bash
python scripts/extract_all_poses.py "data/hiphop_la/pkl" "data/hiphop_la/poses_npy"
```

- Programmatic single-file extraction:

```python
from src.data.extract import extract_poses
out = extract_poses("data/hiphop_la/pkl/<file>.pkl", "data/hiphop_la/poses_npy")
```

## License
Add a license if applicable.
