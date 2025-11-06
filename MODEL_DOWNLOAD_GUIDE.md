# Model Download Guide

## Important: Models are NOT in Git Repository

The model files (`*.pth`) are **too large** for GitHub and are **NOT stored in the Git repository**.

### Model Files (Excluded from Git)
- `video_depth_anything_vitl.pth` (1.4GB)
- `metric_video_depth_anything_vitl.pth` (1.4GB)
- `video_depth_anything_vits.pth` (111MB)
- `video_depth_anything_vitb.pth` (~600MB)

## How Models Are Downloaded

### In Google Colab (Automatic)
The Colab template automatically downloads models from **HuggingFace Hub**:
1. Checks Drive cache first (for faster subsequent runs)
2. Downloads from HuggingFace if not cached
3. Saves to Drive for future use

**Repository**: `depth-anything/Video-Depth-Anything-Large`

### Manual Download (Local Use)
If you need models locally, download from HuggingFace:

```python
from huggingface_hub import hf_hub_download

# Download vitl model
model_path = hf_hub_download(
    repo_id="depth-anything/Video-Depth-Anything-Large",
    filename="video_depth_anything_vitl.pth",
    local_dir="./models"
)

# Download metric vitl model
metric_model_path = hf_hub_download(
    repo_id="depth-anything/Video-Depth-Anything-Large",
    filename="metric_video_depth_anything_vitl.pth",
    local_dir="./models"
)
```

### Available Models
- `video_depth_anything_vits.pth` - Small model
- `video_depth_anything_vitb.pth` - Medium model
- `video_depth_anything_vitl.pth` - Large model (default)
- `metric_video_depth_anything_vitl.pth` - Metric depth version

## Git Configuration

The `.gitignore` file excludes model files:
- `models/` folder in root
- `external_engine/models/` folder
- All `*.pth` files

**Never commit model files to Git!**

