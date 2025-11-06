#!/usr/bin/env python3
"""
Colab-optimized Video Depth Anything Processor
Designed specifically for Google Colab batch processing with Drive integration
"""
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import OpenEXR
import Imath
from pathlib import Path
from tqdm import tqdm
import shutil

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import save_video


class ColabVDAProcessor:
    """Colab-optimized processor for Video Depth Anything"""
    
    def __init__(self, model_path, encoder='vitl', metric=False, device='cuda'):
        """
        Initialize the processor
        
        Args:
            model_path: Path to model checkpoint
            encoder: Model encoder size ('vits', 'vitb', 'vitl')
            metric: Whether to use metric depth model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.encoder = encoder
        self.metric = metric
        
        # Model configuration
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        # Load model
        print(f"Loading model: {encoder}, metric={metric}")
        self.model = VideoDepthAnything(**model_configs[encoder], metric=metric)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        self.model = self.model.to(device).eval()
        print("✅ Model loaded successfully")
    
    def process_sequence(
        self,
        frames,
        exr_output_dir,
        depth_mp4_dir=None,
        first_frame=1001,
        fps=24,
        create_depth_vis_mp4=True
    ):
        """
        Process a sequence of frames and save EXR + MP4
        
        Args:
            frames: List of numpy arrays (frames)
            exr_output_dir: Directory to save EXR files
            depth_mp4_dir: Directory to save MP4 preview (optional)
            first_frame: First frame number for naming
            fps: Frames per second for MP4
            create_depth_vis_mp4: Whether to create MP4 preview
            
        Returns:
            dict with processing results
        """
        print(f"Processing {len(frames)} frames...")
        
        # Convert frames to numpy array if needed
        # VideoDepthAnything expects frames as numpy array with shape [T, H, W, C]
        if isinstance(frames, list):
            frames_array = np.array(frames)
        else:
            frames_array = frames
        
        # Ensure frames are in correct format [T, H, W, C] where T is number of frames
        if len(frames_array.shape) == 4:
            # Already in [T, H, W, C] format
            pass
        elif len(frames_array.shape) == 3:
            # Single frame [H, W, C], add time dimension
            frames_array = frames_array[np.newaxis, :]
        else:
            raise ValueError(f"Unexpected frames shape: {frames_array.shape}")
        
        # Process depth maps
        print("Running depth estimation...")
        with torch.no_grad():
            depths, _ = self.model.infer_video_depth(
                frames_array,
                target_fps=fps,
                input_size=518,
                device=self.device,
                fp32=False
            )
        
        # Convert to numpy if needed
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        
        print(f"✅ Generated {len(depths)} depth maps")
        
        # Save EXR files
        os.makedirs(exr_output_dir, exist_ok=True)
        print(f"Saving EXR files to: {exr_output_dir}")
        
        for i, depth in enumerate(tqdm(depths, desc="Saving EXR")):
            frame_num = first_frame + i
            output_exr = os.path.join(exr_output_dir, f"frame_{frame_num:05d}.exr")
            
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()
        
        print(f"✅ Saved {len(depths)} EXR files")
        
        # Create MP4 preview if requested
        if create_depth_vis_mp4 and depth_mp4_dir:
            os.makedirs(depth_mp4_dir, exist_ok=True)
            print(f"Creating MP4 preview in: {depth_mp4_dir}")
            
            try:
                # Use original frames and depths to create visualization
                depth_vis_path = os.path.join(depth_mp4_dir, "depth_vis.mp4")
                save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=False)
                print(f"✅ Saved MP4 preview: {depth_vis_path}")
            except Exception as e:
                print(f"⚠️  MP4 creation failed: {e}")
        
        return {
            "status": "success",
            "exr_output_dir": exr_output_dir,
            "depth_mp4_dir": depth_mp4_dir if create_depth_vis_mp4 else None,
            "depth_frames": len(depths),
            "fps": fps
        }


def process_batch_sequences(config, model_path, encoder='vitl', metric=False, device='cuda'):
    """
    Process multiple sequences in batch mode
    
    Args:
        config: Batch configuration dictionary
        model_path: Path to model checkpoint
        encoder: Model encoder size
        metric: Whether to use metric depth model
        device: Device to run on
        
    Returns:
        List of processing results
    """
    # Initialize processor (model loaded once)
    processor = ColabVDAProcessor(model_path, encoder, metric, device)
    
    results = []
    sequences = config.get('sequences', [])
    shared_settings = config.get('shared_settings', {})
    
    for idx, seq_config in enumerate(sequences, 1):
        shot_name = seq_config.get('shot_name', f'seq_{idx}')
        print(f"\n{'='*60}")
        print(f"Processing Sequence {idx}/{len(sequences)}: {shot_name}")
        print(f"{'='*60}")
        
        # Get sequence parameters
        input_pattern = seq_config.get('input_video', '')
        first_frame = seq_config.get('first_frame', 1001)
        last_frame = seq_config.get('last_frame', 1100)
        exr_output_dir = seq_config.get('exr_output_dir', f'/content/output/seq_{idx}_depth_exr')
        depth_mp4_dir = seq_config.get('depth_mp4_dir', f'/content/output/seq_{idx}_depth_mp4')
        
        # Load frames from Drive
        print(f"Loading frames: {first_frame}-{last_frame}")
        frames = []
        for frame_num in range(first_frame, last_frame + 1):
            frame_path = input_pattern % frame_num
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                frames.append(np.array(img))
            else:
                print(f"⚠️  Missing frame: {frame_path}")
        
        print(f"Loaded {len(frames)} frames")
        
        # Process sequence
        result = processor.process_sequence(
            frames=frames,
            exr_output_dir=exr_output_dir,
            depth_mp4_dir=depth_mp4_dir if shared_settings.get('create_depth_vis_mp4', True) else None,
            first_frame=first_frame,
            fps=24,  # Default FPS
            create_depth_vis_mp4=shared_settings.get('create_depth_vis_mp4', True)
        )
        
        result['shot_name'] = shot_name
        results.append(result)
    
    return results

