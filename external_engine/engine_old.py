#!/usr/bin/env python3
"""
Simple Video Depth Anything Engine - Memory Optimized
Follows the original Video-Depth-Anything processing pipeline with memory optimization
for UHD/4K processing through intelligent half-resolution scaling
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import torch
    TORCH_AVAILABLE = True
    print("[INFO] PyTorch successfully imported")
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Engine will not work.")

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import save_video

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleVideoDepthEngine:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or str(current_dir / "models")
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot initialize engine")
            return
        
        # Simple device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info(f"Using CUDA device")
        else:
            self.device = 'cpu'
            logger.info(f"Using CPU device")
        
    def load_model(self, encoder='vitl', metric=False):
        """Load the Video Depth Anything model - SIMPLE VERSION"""
        try:
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
            checkpoint_path = os.path.join(self.model_path, f"{checkpoint_name}_{encoder}.pth")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            self.model = VideoDepthAnything(**model_configs[encoder], metric=metric)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
            self.model = self.model.to(self.device).eval()
            
            logger.info(f"Model loaded successfully: {checkpoint_name}_{encoder}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def read_frames(self, input_path, job_data):
        """Read frames from image sequence with memory optimization"""
        try:
            import cv2
            
            # Get frame range (ensure integers)
            start_frame = int(job_data.get('start_frame', 1))
            end_frame = int(job_data.get('end_frame', 100))
            
            # Get memory optimization settings
            max_res = job_data.get('max_res', 1920)  # Default to 1920 for memory optimization
            enable_half_res = job_data.get('enable_half_res', True)  # Default to True for UHD
            
            print(f"[LOADING] Loading image sequence...")
            print(f"[FRAMES] Frame range: {start_frame} to {end_frame}")
            print(f"[MEMORY] Max resolution: {max_res}px (half-res scaling: {'ON' if enable_half_res else 'OFF'})")
            
            frames = []
            original_h, original_w = None, None
            
            for frame_num in range(start_frame, end_frame + 1):
                # Replace frame padding with actual frame number
                frame_path = input_path.replace("%04d", f"{frame_num:04d}")
                frame_path = frame_path.replace("####", f"{frame_num:04d}")
                
                if os.path.exists(frame_path) and os.path.isfile(frame_path):
                    # Read image
                    img = cv2.imread(frame_path)
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Store original dimensions from first frame
                        if original_h is None:
                            original_h, original_w = img.shape[:2]
                            print(f"[DIMENSIONS] Original: {original_w}x{original_h}")
                        
                        # Apply memory optimization scaling
                        if enable_half_res and max_res > 0 and max(original_h, original_w) > max_res:
                            scale = max_res / max(original_h, original_w)
                            new_h = int(original_h * scale)
                            new_w = int(original_w * scale)
                            
                            # Ensure even dimensions for video encoding
                            new_h = new_h if new_h % 2 == 0 else new_h + 1
                            new_w = new_w if new_w % 2 == 0 else new_w + 1
                            
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            
                            if frame_num == start_frame:  # Log only once
                                print(f"[SCALING] Applied scaling: {original_w}x{original_h} → {new_w}x{new_h} (scale: {scale:.3f})")
                                print(f"[MEMORY] Memory usage reduced by ~{(1-scale**2)*100:.1f}%")
                        
                        frames.append(img)
                        logger.info(f"Loaded frame {frame_num}")
                    else:
                        logger.warning(f"Could not read frame: {frame_path}")
                else:
                    logger.warning(f"Frame not found: {frame_path}")
            
            if len(frames) == 0:
                logger.error(f"No frames loaded from sequence: {input_path}")
                return [], 24.0
            
            # Convert to numpy array
            frames = np.array(frames)
            logger.info(f"Successfully loaded {len(frames)} frames")
            
            # Store dimensions (original and processed)
            if len(frames) > 0:
                processed_h, processed_w = frames[0].shape[:2]
                job_data['original_width'] = original_w
                job_data['original_height'] = original_h
                job_data['processed_width'] = processed_w
                job_data['processed_height'] = processed_h
                logger.info(f"Original dimensions: {original_w}x{original_h}")
                logger.info(f"Processed dimensions: {processed_w}x{processed_h}")
            
            return frames, 24.0
            
        except Exception as e:
            logger.error(f"Failed to read frames: {str(e)}")
            return [], 24.0
    
    def save_depth_sequence(self, depths, output_path, base_name, job_data=None):
        """Save depth frames as EXR or PNG based on output_format parameter"""
        try:
            # Get output format preference first
            output_format = job_data.get('output_format', 'png') if job_data else 'png'  # Default to PNG for UHD reliability
            
            # Create output directory
            if output_path.endswith('.exr'):
                depth_dir = os.path.dirname(output_path)
                depth_dir = os.path.join(depth_dir, f"depth_{output_format}")  # Use actual format
            else:
                depth_dir = os.path.join(output_path, f"depth_{output_format}")  # Use actual format
            
            os.makedirs(depth_dir, exist_ok=True)
            print(f"Creating output directory: {depth_dir}")
            
            # Get frame range (ensure integers)
            start_frame = int(job_data.get('start_frame', 1)) if job_data else 1
            
            # Get depth normalization settings from Nuke GUI
            enable_normalization = job_data.get('enable_normalization', False) if job_data else False
            near_value = job_data.get('near_value', 0.0) if job_data else 0.0
            far_value = job_data.get('far_value', 5.0) if job_data else 5.0
            invert_depth = job_data.get('invert_depth', False) if job_data else False
            
            
            if enable_normalization:
                print(f"[NORMALIZATION] Depth normalization: ENABLED - Near={near_value}, Far={far_value}, Invert={invert_depth}")
            else:
                print(f"[NORMALIZATION] Depth normalization: DISABLED - Using raw depth values (like original)")
            
            print(f"[SAVING] Saving {len(depths)} depth frames...")
            print(f"[FORMAT] Using {output_format.upper()} format")
            if output_format.lower() == 'exr':
                print(f"[FORMAT] EXR format - high precision 32-bit float depth data")
            else:
                print(f"[FORMAT] PNG format - 8-bit normalized depth data")
            
            # Save each depth frame
            for i in range(len(depths)):
                depth = depths[i]
                # Progress indicator
                progress = (i + 1) / len(depths) * 100
                print(f"[PROGRESS] {progress:.1f}% - Processing frame {i+1}/{len(depths)}")
                frame_number = start_frame + i
                depth_file = os.path.join(depth_dir, f"{base_name}_depth.{frame_number:04d}.{output_format}")
                
                print(f"[SAVING] Frame {frame_number}: {depth_file}")
                
                # Ensure depth is numpy array
                if not isinstance(depth, np.ndarray):
                    depth = np.array(depth)
                
                # Apply depth normalization if enabled
                if enable_normalization:
                    depth_normalized = self.normalize_depth(depth, near_value, far_value, invert_depth)
                else:
                    # Use raw depth values like original implementation
                    depth_normalized = depth
                
                # Save based on output format (restored original working implementation)
                if output_format.lower() == 'exr':
                    # Save as EXR for high precision (32-bit float) - ORIGINAL WORKING METHOD
                    try:
                        import OpenEXR
                        import Imath
                        
                        # Convert to float32 if needed
                        depth_float = depth_normalized.astype(np.float32)
                        
                        header = OpenEXR.Header(depth_float.shape[1], depth_float.shape[0])
                        header["channels"] = {
                            "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                        }
                        
                        exr_file = OpenEXR.OutputFile(depth_file, header)
                        exr_file.writePixels({"Z": depth_float.tobytes()})
                        exr_file.close()
                        
                        print(f"[SUCCESS] Saved frame {frame_number} as EXR")
                        
                    except Exception as e:
                        print(f"[ERROR] EXR failed for frame {frame_number}: {str(e)}")
                        continue
                else:
                    # Save as PNG
                    try:
                        import cv2
                        # Normalize depth to 0-255 range for PNG
                        depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
                        if depth_max > depth_min:
                            depth_png = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                        else:
                            depth_png = np.zeros_like(depth_normalized, dtype=np.uint8)
                        
                        cv2.imwrite(depth_file, depth_png)
                        print(f"[SUCCESS] Saved frame {frame_number} as PNG")
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to save PNG frame {frame_number}: {str(e)}")
                        continue
            
            print(f"[SUCCESS] Saved {len(depths)} depth frames to {depth_dir}")
            logger.info(f"Saved {len(depths)} depth frames to {depth_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save depth sequence: {str(e)}")
            print(f"[CRITICAL ERROR] Failed to save depth sequence: {str(e)}")
            import traceback
            print(f"[CRITICAL ERROR] Traceback: {traceback.format_exc()}")
            # Don't re-raise the exception, just log it and continue
    
    def normalize_depth(self, depth, near_value, far_value, invert_depth):
        """Normalize depth values for VFX production - Nuke GUI compatibility"""
        try:
            # Convert to numpy if needed
            if not isinstance(depth, np.ndarray):
                depth = np.array(depth)
            
            # Normalize depth to 0-1 range based on near/far values
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max > depth_min:
                # Normalize to 0-1
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
                
                # Apply near/far clipping
                if far_value > near_value:
                    # Map to near/far range
                    depth_normalized = near_value + (far_value - near_value) * depth_normalized
                else:
                    # Invert the mapping
                    depth_normalized = far_value + (near_value - far_value) * depth_normalized
            else:
                # Constant depth
                depth_normalized = np.full_like(depth, near_value)
            
            # Apply inversion if requested
            if invert_depth:
                depth_normalized = far_value - (depth_normalized - near_value)
            
            return depth_normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize depth: {str(e)}")
            return depth
    
    
    def _create_mp4_files(self, frames, depths, output_path, input_name, fps):
        """Create MP4 files for source video and depth visualization"""
        try:
            # Create depth_mp4 subdirectory for MP4 files
            if output_path.endswith('.exr'):
                base_output_dir = os.path.dirname(output_path)
            else:
                base_output_dir = output_path
            
            depth_mp4_dir = os.path.join(base_output_dir, "depth_mp4")
            os.makedirs(depth_mp4_dir, exist_ok=True)
            
            print(f"[MP4] Creating MP4 files in: {depth_mp4_dir}")
            logger.info(f"Creating MP4 files in: {depth_mp4_dir}")
            
            # Save source video (like original implementation)
            source_video = os.path.join(depth_mp4_dir, f"{input_name}_src.mp4")
            print(f"[MP4] Creating source video: {source_video}")
            
            # Convert frames to numpy array if it's a list
            if isinstance(frames, list):
                frames = np.array(frames)
            
            # Use processed frames (may be scaled down)
            if len(frames) > 0:
                current_height, current_width = frames[0].shape[:2]
                print(f"[MP4] Using processed frame resolution: {current_width}x{current_height}")
            
            save_video(frames, source_video, fps=fps, is_depths=False)
            print(f"[MP4] Source video created: {source_video}")
            
            # Save depth visualization video (like original implementation)
            depth_video = os.path.join(depth_mp4_dir, f"{input_name}_vis.mp4")
            print(f"[MP4] Creating depth visualization: {depth_video}")
            
            # Convert depths to numpy array if it's a list
            if isinstance(depths, list):
                depths = np.array(depths)
            
            save_video(depths, depth_video, fps=fps, is_depths=True)
            print(f"[MP4] Depth visualization created: {depth_video}")
            
            logger.info(f"MP4 files created successfully: {source_video}, {depth_video}")
            
        except Exception as e:
            logger.error(f"Failed to create MP4 files: {str(e)}")
            print(f"[ERROR] Failed to create MP4 files: {str(e)}")
    
    def process_video(self, job_data):
        """Process video - SIMPLE VERSION using original method"""
        try:
            input_path = job_data["input_path"]
            output_path = job_data["output_path"]
            encoder = job_data.get("encoder", "vitl")
            metric = job_data.get("metric", False)
            fp32_mode = job_data.get('fp32', False)
            
            logger.info(f"Processing video: {input_path}")
            logger.info(f"Output path: {output_path}")
            
            # Load model if not already loaded
            if self.model is None:
                if not self.load_model(encoder, metric):
                    return {"status": "error", "message": "Failed to load model"}
            
            # Read frames
            frames, fps = self.read_frames(input_path, job_data)
            if len(frames) == 0:
                return {"status": "error", "message": "No frames loaded"}
            
            logger.info(f"Loaded {len(frames)} frames")
            
            # Process using ORIGINAL Video-Depth-Anything method
            print(f"[PROCESSING] Using ORIGINAL Video-Depth-Anything method...")
            depths, fps = self.model.infer_video_depth(
                frames, 
                fps, 
                input_size=518,  # Original default
                device=self.device, 
                fp32=fp32_mode
            )
            
            logger.info(f"Processing completed: {len(depths)} depth maps generated")
            
            # Model returns depth maps at processed resolution
            if len(depths) > 0:
                current_height, current_width = depths[0].shape
                original_width = job_data.get('original_width', current_width)
                original_height = job_data.get('original_height', current_height)
                processed_width = job_data.get('processed_width', current_width)
                processed_height = job_data.get('processed_height', current_height)
                
                print(f"[OUTPUT] Depth maps at processed resolution: {current_width}x{current_height}")
                logger.info(f"Depth maps at processed resolution: {current_width}x{current_height}")
                
                # Check if we need to upscale back to original resolution
                if current_width != original_width or current_height != original_height:
                    print(f"[UPSCALING] Upscaling depth maps from {current_width}x{current_height} to {original_width}x{original_height}")
                    logger.info(f"Upscaling depth maps to original resolution: {original_width}x{original_height}")
                    
                    # Upscale depth maps back to original resolution
                    import cv2
                    upscaled_depths = []
                    for depth in depths:
                        upscaled_depth = cv2.resize(depth, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
                        upscaled_depths.append(upscaled_depth)
                    depths = np.array(upscaled_depths)
                    print(f"[UPSCALING] Upscaling completed: {len(depths)} depth maps")
                else:
                    print(f"[OUTPUT] No upscaling needed - already at original resolution")
            
            # Generate output filename
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            import re
            input_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_name)
            input_name = input_name.strip('_.')
            
            # Save depth sequence
            self.save_depth_sequence(depths, output_path, input_name, job_data)
            
            # Create MP4 files (like original implementation)
            self._create_mp4_files(frames, depths, output_path, input_name, fps)
            
            logger.info(f"Processing completed successfully: {output_path}")
            return {
                "status": "success", 
                "output_path": output_path,
                "depth_frames": len(depths),
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {"status": "error", "message": str(e)}

def main():
    """Main engine loop - SIMPLE VERSION"""
    if not TORCH_AVAILABLE:
        print("=" * 60)
        print("CRITICAL ERROR: PyTorch not available!")
        print("=" * 60)
        return
    
    engine = SimpleVideoDepthEngine()
    
    # Job queue file path
    current_dir = Path(__file__).parent.parent
    job_queue_file = current_dir / "communication" / "job_queue.json"
    status_file = current_dir / "communication" / "status.json"
    
    # Create communication directory
    os.makedirs(current_dir / "communication", exist_ok=True)
    
    print("=" * 60)
    print("SIMPLE VDA ENGINE STARTED")
    print("=" * 60)
    print("Monitoring job queue for new processing requests...")
    print("Press Ctrl+C to stop the engine")
    print("=" * 60)
    
    # Update status to running
    status = {"status": "running", "message": "Simple engine ready"}
    with open(status_file, 'w') as f:
        json.dump(status, f)
    
    logger.info("Simple Video Depth Anything Engine started")
    
    while True:
        try:
            # Check for new jobs
            if os.path.exists(job_queue_file):
                print("=" * 60)
                print("NEW JOB DETECTED!")
                print("=" * 60)
                
                with open(job_queue_file, 'r') as f:
                    job_data = json.load(f)
                
                print(f"Input: {job_data.get('input_path', 'Unknown')}")
                print(f"Output: {job_data.get('output_path', 'Unknown')}")
                print(f"Frames: {job_data.get('start_frame', 1)}-{job_data.get('end_frame', 100)}")
                print(f"Encoder: {job_data.get('encoder', 'vitl')}")
                print("=" * 60)
                print("STARTING PROCESSING...")
                print("=" * 60)
                
                # Update status to processing
                status = {"status": "processing", "message": "Processing video..."}
                with open(status_file, 'w') as f:
                    json.dump(status, f)
                
                # Process the video
                result = engine.process_video(job_data)
                
                # Update status with result
                with open(status_file, 'w') as f:
                    json.dump(result, f)
                
                # Remove job file
                os.remove(job_queue_file)
                
                print("=" * 60)
                print("PROCESSING COMPLETED!")
                print("=" * 60)
                print(f"Status: {result['status']}")
                if result['status'] == 'success':
                    print(f"Output: {result.get('output_path', 'Unknown')}")
                    print(f"Frames processed: {result.get('depth_frames', 0)}")
                    print(f"FPS: {result.get('fps', 24)}")
                else:
                    print(f"Error: {result.get('message', 'Unknown error')}")
                print("=" * 60)
                print("Waiting for next job...")
                print("=" * 60)
                
                logger.info(f"Job completed: {result['status']}")
            
            time.sleep(0.5)  # Check every 500ms
            
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("ENGINE STOPPED BY USER")
            print("=" * 60)
            
            # Update status to stopped
            status = {"status": "stopped", "message": "Engine stopped by user"}
            with open(status_file, 'w') as f:
                json.dump(status, f)
            
            logger.info("Engine stopped by user")
            break
        except Exception as e:
            print(f"\nENGINE ERROR: {str(e)}")
            logger.error(f"Engine error: {str(e)}")
            
            # Update status to error
            status = {"status": "error", "message": str(e)}
            with open(status_file, 'w') as f:
                json.dump(status, f)
            
            time.sleep(1)

if __name__ == "__main__":
    main()
