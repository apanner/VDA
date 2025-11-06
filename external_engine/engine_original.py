#!/usr/bin/env python3
"""
Video Depth Anything Engine - Original Implementation
Matches the sample.md implementation exactly - no memory optimization
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
from utils.dc_utils import read_video_frames, save_video

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('original_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OriginalVideoDepthEngine:
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
        """Load the Video Depth Anything model - ORIGINAL METHOD"""
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
    
    def process_video_original(self, job_data):
        """Process video - EXACTLY like sample.md implementation"""
        try:
            # Extract parameters exactly like sample.md
            input_video = job_data["input_video"]  # sample.md uses "input_video"
            exr_output_dir = job_data["exr_output_dir"]  # sample.md uses "exr_output_dir"
            first_frame = int(job_data['first_frame'])  # sample.md uses "first_frame"
            last_frame = int(job_data['last_frame'])    # sample.md uses "last_frame"
            metric_depth = job_data['metric_depth']     # sample.md uses "metric_depth"
            floating_point = job_data['floating_point'] # sample.md uses "floating_point"
            encoder = job_data.get('encoder', 'vits')   # sample.md defaults to 'vits'
            
            # Default args like sample.md
            max_len = -1  # Maximum length of the input video, -1 means no limit
            target_fps = -1 # Target fps of the input video, -1 means the original fps
            max_res = -1  # Max resolution of the input video, -1 means no limit (ORIGINAL)
            input_size = 518  # Input size
            
            logger.info(f"Processing video: {input_video}")
            logger.info(f"Output directory: {exr_output_dir}")
            logger.info(f"Frame range: {first_frame} to {last_frame}")
            logger.info(f"Max resolution: {max_res} (no limit - ORIGINAL METHOD)")
            
            # Load model exactly like sample.md - FIXED to use correct encoder
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            checkpoint_name = 'metric_video_depth_anything' if metric_depth else 'video_depth_anything'
            
            # Use the checkpoint path from job data like sample.md
            checkpoint_path = job_data[f"{checkpoint_name}_checkpoint"]
            
            # Determine encoder from job data or checkpoint path
            if 'encoder' in job_data:
                encoder = job_data['encoder']
                logger.info(f"Using encoder from job data: {encoder}")
            elif 'vits' in checkpoint_path:
                encoder = 'vits'
            elif 'vitb' in checkpoint_path:
                encoder = 'vitb'
            elif 'vitl' in checkpoint_path:
                encoder = 'vitl'
            else:
                encoder = 'vits'  # default fallback
            
            logger.info(f"Detected encoder from checkpoint: {encoder}")
            model_config = model_configs[encoder]
            
            self.model = VideoDepthAnything(**model_config, metric=metric_depth)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
            self.model = self.model.to(self.device).eval()
            
            # Read frames - handle both video files and image sequences
            print(f"[LOADING] Using ORIGINAL method - no memory optimization")
            if input_video.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Video file - use original method
                frames, target_fps = read_video_frames(input_video, max_len, target_fps, max_res)
            else:
                # Image sequence - read frames manually
                import cv2
                frames = []
                for frame_num in range(first_frame, last_frame + 1):
                    frame_path = input_video.replace("%04d", f"{frame_num:04d}")
                    frame_path = frame_path.replace("####", f"{frame_num:04d}")
                    
                    if os.path.exists(frame_path):
                        img = cv2.imread(frame_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            frames.append(img)
                            print(f"Loaded frame {frame_num}")
                
                frames = np.array(frames)
                target_fps = 24.0
            
            if len(frames) == 0:
                return {"status": "error", "message": "No frames loaded"}
            
            logger.info(f"Loaded {len(frames)} frames at original resolution")
            
            # Process using ORIGINAL Video-Depth-Anything method
            print(f"[PROCESSING] Using ORIGINAL Video-Depth-Anything method...")
            depths, fps = self.model.infer_video_depth(
                frames, 
                target_fps, 
                input_size=input_size, 
                device=self.device, 
                fp32=floating_point == "float32"  # sample.md uses string comparison
            )
            
            logger.info(f"Processing completed: {len(depths)} depth maps generated")
            
            # Create output directory
            os.makedirs(exr_output_dir, exist_ok=True)
            print(f"Creating output directory: {exr_output_dir}")
            
            # Save EXR files exactly like sample.md
            frame_range = range(first_frame, last_frame + 1)
            print(f"[SAVING] Saving {len(depths)} EXR files...")
            
            for i, depth in zip(frame_range, depths):
                output_exr = f"{exr_output_dir}/video_depth_{i}.exr"
                if i % 20 == 0:
                    print(f"Writing file: {output_exr}", flush=True)
                
                # EXR saving exactly like sample.md
                import OpenEXR
                import Imath
                
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()
            
            print(f"EXRs successfully written to: {exr_output_dir}")
            logger.info(f"Processing completed successfully: {exr_output_dir}")
            
            return {
                "status": "success", 
                "output_path": exr_output_dir,
                "depth_frames": len(depths),
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {"status": "error", "message": str(e)}

def main():
    """Main engine loop - ORIGINAL VERSION with command line support like sample.md"""
    if not TORCH_AVAILABLE:
        print("=" * 60)
        print("CRITICAL ERROR: PyTorch not available!")
        print("=" * 60)
        return
    
    # Check for command line arguments like sample.md
    if len(sys.argv) > 1:
        # Command line mode - process single job file like sample.md
        job_file = sys.argv[1]
        print("=" * 60)
        print("ORIGINAL VDA ENGINE - COMMAND LINE MODE")
        print("=" * 60)
        print(f"Processing job file: {job_file}")
        print("=" * 60)
        
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            engine = OriginalVideoDepthEngine()
            result = engine.process_video_original(job_data)
            
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
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return
    else:
        # Job queue mode (original behavior)
        engine = OriginalVideoDepthEngine()
        
        # Job queue file path
        current_dir = Path(__file__).parent.parent
        job_queue_file = current_dir / "communication" / "job_queue.json"
        status_file = current_dir / "communication" / "status.json"
    
    # Create communication directory
    os.makedirs(current_dir / "communication", exist_ok=True)
    
    print("=" * 60)
    print("ORIGINAL VDA ENGINE STARTED")
    print("=" * 60)
    print("Using ORIGINAL implementation - no memory optimization")
    print("Monitoring job queue for new processing requests...")
    print("Press Ctrl+C to stop the engine")
    print("=" * 60)
    
    # Update status to running
    status = {"status": "running", "message": "Original engine ready"}
    with open(status_file, 'w') as f:
        json.dump(status, f)
    
    logger.info("Original Video Depth Anything Engine started")
    
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
                result = engine.process_video_original(job_data)
                
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
