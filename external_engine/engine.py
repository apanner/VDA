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

# Enable OpenEXR support for OpenCV (like iMatte does)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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

# Import EXR to ProRes converter
try:
    # Add current directory to path for EXR converter import
    import sys
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Try importing from current directory first
    try:
        from exr_to_prores_converter import convert_exr_to_prores_for_depth, is_exr_sequence
    except ImportError:
        # If that fails, try importing from parent directory
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from exr_to_prores_converter import convert_exr_to_prores_for_depth, is_exr_sequence
    
    EXR_CONVERTER_AVAILABLE = True
    print("[INFO] EXR to ProRes converter available")
except ImportError as e:
    EXR_CONVERTER_AVAILABLE = False
    print(f"WARNING: EXR to ProRes converter not available: {e}")

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
        print(f"[DEBUG] Initializing OriginalVideoDepthEngine...")
        self.model = None
        self.model_path = model_path or str(current_dir / "models")
        
        if not TORCH_AVAILABLE:
            print(f"[ERROR] PyTorch not available - cannot initialize engine")
            logger.error("PyTorch not available - cannot initialize engine")
            return
        
        # Simple device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"[DEBUG] Using CUDA device")
            logger.info(f"Using CUDA device")
        else:
            self.device = 'cpu'
            print(f"[DEBUG] Using CPU device")
            logger.info(f"Using CPU device")
        
        print(f"[DEBUG] Engine initialization completed")
        
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
            print(f"[DEBUG] Starting job processing...")
            
            # Extract parameters exactly like sample.md
            input_video = job_data["input_video"]  # sample.md uses "input_video"
            exr_output_dir = job_data["exr_output_dir"]  # sample.md uses "exr_output_dir"
            first_frame = int(job_data['first_frame'])  # sample.md uses "first_frame"
            last_frame = int(job_data['last_frame'])    # sample.md uses "last_frame"
            metric_depth = job_data['metric_depth']     # sample.md uses "metric_depth"
            floating_point = job_data['floating_point'] # sample.md uses "floating_point"
            encoder = job_data.get('encoder', 'vits')   # sample.md defaults to 'vits'
            
            print(f"[DEBUG] Parameters extracted successfully")
            
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
            
            print(f"[DEBUG] Model loaded and moved to device successfully")
            
            # Read frames - handle both video files and image sequences (including EXR)
            print(f"[LOADING] Using ORIGINAL method - no memory optimization")
            
            # Check if input is EXR sequence and convert to ProRes if needed
            temp_prores_file = None
            actual_input_video = input_video
            
            print(f"[DEBUG] EXR_CONVERTER_AVAILABLE: {EXR_CONVERTER_AVAILABLE}")
            print(f"[DEBUG] Input video path: {input_video}")
            
            if EXR_CONVERTER_AVAILABLE:
                is_exr = is_exr_sequence(input_video)
                print(f"[DEBUG] is_exr_sequence result: {is_exr}")
                
                if is_exr:
                    print(f"[EXR->ProRes] Detected EXR sequence input, converting to Apple ProRes...")
                    temp_prores_file = convert_exr_to_prores_for_depth(
                        input_video, first_frame, last_frame, target_fps if target_fps > 0 else 24.0
                    )
                    
                    if temp_prores_file:
                        actual_input_video = temp_prores_file
                        print(f"[EXR->ProRes] Successfully converted EXR to ProRes: {temp_prores_file}")
                    else:
                        print(f"[ERROR] Failed to convert EXR sequence to ProRes")
                        return {"status": "error", "message": "Failed to convert EXR sequence to ProRes"}
                else:
                    print(f"[DEBUG] Input is not an EXR sequence, proceeding with regular processing")
            else:
                print(f"[DEBUG] EXR converter not available, proceeding with regular processing")
            
            if actual_input_video.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Video file - use original method (now supports ProRes from EXR conversion)
                frames, target_fps = read_video_frames(actual_input_video, max_len, target_fps, max_res)
            else:
                # Image sequence - read frames manually (supports JPG, PNG, etc.)
                import cv2
                frames = []
                print(f"[DEBUG] Loading image sequence from {first_frame} to {last_frame}")
                
                for frame_num in range(first_frame, last_frame + 1):
                    # Try multiple frame number formats
                    frame_path = actual_input_video.replace("%04d", f"{frame_num:04d}")
                    frame_path = frame_path.replace("####", f"{frame_num:04d}")
                    frame_path = frame_path.replace("%d", f"{frame_num}")
                    
                    print(f"[DEBUG] Trying frame path: {frame_path}")
                    
                    if os.path.exists(frame_path):
                        print(f"[DEBUG] Found frame: {frame_path}")
                        # Check if it's an EXR file
                        if frame_path.lower().endswith('.exr'):
                            # Read EXR file using OpenEXR
                            img = self._read_exr_frame(frame_path)
                            if img is not None:
                                frames.append(img)
                                print(f"Loaded frame {frame_num} (EXR)")
                        else:
                            # Regular image file (JPG, PNG, TIFF, etc.)
                            img = cv2.imread(frame_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                frames.append(img)
                                print(f"Loaded frame {frame_num} (Image)")
                            else:
                                print(f"[ERROR] Failed to load image: {frame_path}")
                    else:
                        print(f"[WARNING] Frame not found: {frame_path}")
                
                if len(frames) == 0:
                    print(f"[ERROR] No frames loaded! Check input path: {actual_input_video}")
                    print(f"[ERROR] Tried frame range: {first_frame} to {last_frame}")
                    return {"status": "error", "message": f"No frames loaded from {actual_input_video}"}
                
                frames = np.array(frames)
                target_fps = 24.0
                print(f"[SUCCESS] Loaded {len(frames)} frames successfully")
            
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
            
            # Apply depth normalization if enabled
            enable_normalization = job_data.get('enable_normalization', False)
            if enable_normalization:
                near_value = job_data.get('near_value', 0.0)
                far_value = job_data.get('far_value', 5.0)
                invert_depth = job_data.get('invert_depth', False)
                
                print(f"[NORMALIZATION] Applying depth normalization:")
                print(f"  - Near value: {near_value}")
                print(f"  - Far value: {far_value}")
                print(f"  - Invert depth: {invert_depth}")
                
                # Apply normalization to each depth frame
                for i, depth in enumerate(depths):
                    depths[i] = self.normalize_depth(depth, near_value, far_value, invert_depth)
                    if i % 10 == 0 or i < 3:
                        print(f"[NORMALIZATION] Processed frame {i+1}/{len(depths)}")
                
                print(f"[NORMALIZATION] Depth normalization completed for {len(depths)} frames")
            else:
                print(f"[NORMALIZATION] Depth normalization disabled - using raw depth values")
            
            # Debug: Check depths array
            print(f"[DEBUG] Depths array info:")
            print(f"  - Length: {len(depths)}")
            if len(depths) > 0:
                print(f"  - First depth shape: {depths[0].shape}")
                print(f"  - First depth dtype: {depths[0].dtype}")
                print(f"  - First depth range: {depths[0].min():.3f} - {depths[0].max():.3f}")
            else:
                print(f"  - ERROR: Depths array is empty!")
                return {"status": "error", "message": "No depth maps were generated"}
            
            # Create output directory exactly like sample.md
            os.makedirs(exr_output_dir, exist_ok=True)
            print(f"Creating output directory: {exr_output_dir}")
            
            # Save EXR files with shot name
            # Use EXR frame numbers if provided, otherwise use input frame range
            exr_first_frame = job_data.get('exr_first_frame', first_frame)
            exr_last_frame = job_data.get('exr_last_frame', last_frame)
            # Use the actual number of depth maps available
            actual_frame_count = len(depths)
            frame_range = range(exr_first_frame, exr_first_frame + actual_frame_count)
            print(f"[DEBUG] Adjusted frame range: {frame_range.start} to {frame_range.stop-1} ({actual_frame_count} frames)")
            print(f"[SAVING] Saving {len(depths)} EXR files...")
            print(f"[DEBUG] EXR frame range: {exr_first_frame} to {exr_last_frame}")
            
            # Extract shot name from input video path
            import re
            input_base_name = os.path.splitext(os.path.basename(input_video))[0]
            # Remove frame patterns like %04d, ####, etc.
            input_base_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_base_name)
            input_base_name = input_base_name.strip('_.')
            
            print(f"[DEBUG] Using shot name: {input_base_name}")
            
            saved_count = 0
            failed_count = 0
            
            print(f"[EXR] Starting EXR creation for {len(depths)} frames...")
            print(f"[EXR] Frame range: {frame_range.start} to {frame_range.stop-1}")
            print(f"[DEBUG] Input base name: {input_base_name}")
            print(f"[DEBUG] EXR output directory: {exr_output_dir}")
            print(f"[DEBUG] First few output paths will be:")
            for i in range(min(3, len(depths))):
                frame_num = frame_range.start + i
                sample_path = f"{exr_output_dir}/{input_base_name}.{frame_num}.exr"
                print(f"[DEBUG]   Frame {frame_num}: {sample_path}")
            
            for frame_idx, (i, depth) in enumerate(zip(frame_range, depths)):
                output_exr = f"{exr_output_dir}/{input_base_name}.{i}.exr"
                
                # More frequent progress reporting to prevent hanging
                if frame_idx % 5 == 0 or frame_idx < 3:
                    print(f"[EXR] Frame {frame_idx+1}/{len(depths)}: Processing frame {i}", flush=True)
                
                try:
                    # Use OpenCV with OpenEXR support (like iMatte does)
                    # Convert depth to float32 for EXR (values 0-1)
                    exr_depth = depth.astype("float32")
                    
                    # Import cv2 here to ensure OpenEXR environment is set
                    import cv2
                    cv2.imwrite(output_exr, exr_depth)
                    saved_count += 1
                    
                    # More frequent progress updates
                    if saved_count % 10 == 0 or saved_count <= 5:
                        print(f"[PROGRESS] Saved {saved_count}/{len(depths)} EXR files...", flush=True)
                    
                    # Verify file was actually created
                    if not os.path.exists(output_exr):
                        raise Exception(f"EXR file was not created: {output_exr}")
                        
                except cv2.error as cv_err:
                    if "OpenEXR" in str(cv_err):
                        print(f"[EXR] OpenEXR not available, saving as PNG instead")
                        png_path = output_exr.replace('.exr', '.png')
                        # Convert depth to 0-255 range for PNG
                        depth_png = ((depth - depth.min()) / 
                                    (depth.max() - depth.min()) * 255).astype(np.uint8)
                        cv2.imwrite(png_path, depth_png)
                        saved_count += 1
                    else:
                        failed_count += 1
                        print(f"[ERROR] Failed to save EXR for frame {i}: {str(cv_err)}")
                        print(f"[ERROR] Output path: {output_exr}")
                        print(f"[ERROR] Depth shape: {depth.shape}, dtype: {depth.dtype}")
                        continue
                except Exception as e:
                    failed_count += 1
                    print(f"[ERROR] Failed to save EXR for frame {i}: {str(e)}")
                    print(f"[ERROR] Output path: {output_exr}")
                    print(f"[ERROR] Depth shape: {depth.shape}, dtype: {depth.dtype}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Report EXR saving results
            print(f"[EXR RESULTS] Successfully saved: {saved_count} EXR files")
            if failed_count > 0:
                print(f"[EXR RESULTS] Failed to save: {failed_count} EXR files")
                if saved_count == 0:
                    print(f"[ERROR] No EXR files were created! Check the error messages above.")
                    return {"status": "error", "message": f"Failed to create any EXR files. {failed_count} failures."}
                else:
                    print(f"[WARNING] Some EXR files failed, but {saved_count} were saved successfully")
            else:
                print(f"[SUCCESS] All {saved_count} EXR files created successfully!")
            
            print(f"[COMPLETE] EXR creation finished! Files written to: {exr_output_dir}")
            logger.info(f"EXR processing completed successfully: {exr_output_dir}")
            
            # Create MP4 files AFTER EXR saving is complete (separate step)
            print(f"[MP4] Starting MP4 creation after successful EXR saving...")
            try:
                # Generate input name for MP4 files
                input_name = os.path.splitext(os.path.basename(input_video))[0]
                import re
                input_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_name)
                input_name = input_name.strip('_.')
                
                # Use optimized MP4 creation (create both source and depth vis by default)
                self._create_mp4_files(
                    frames, depths, exr_output_dir, input_name, fps,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    job_data=job_data,
                    create_source_mp4=True,
                    create_depth_vis_mp4=True
                )
                print(f"[MP4] MP4 files created successfully")
            except Exception as e:
                print(f"[WARNING] MP4 creation failed, but EXR files were saved successfully: {str(e)}")
                # Don't fail the entire process if MP4 creation fails
            
            # Clean up temporary ProRes file if it was created
            if temp_prores_file and os.path.exists(temp_prores_file):
                try:
                    os.remove(temp_prores_file)
                    print(f"[EXR->ProRes] Cleaned up temporary ProRes file: {temp_prores_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary ProRes file: {str(e)}")
            
            return {
                "status": "success", 
                "output_path": depth_exr_dir,
                "depth_frames": len(depths),
                "fps": fps
            }
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            logger.error(f"Processing failed: {str(e)}")
            
            # Clean up temporary ProRes file if it was created
            if 'temp_prores_file' in locals() and temp_prores_file and os.path.exists(temp_prores_file):
                try:
                    os.remove(temp_prores_file)
                    print(f"[EXR->ProRes] Cleaned up temporary ProRes file after error: {temp_prores_file}")
                except Exception as cleanup_e:
                    logger.warning(f"Failed to clean up temporary ProRes file after error: {str(cleanup_e)}")
            
            return {"status": "error", "message": str(e)}
    
    def process_video_optimized(self, job_data, progress_task=None):
        """
        Simple video processing workflow like sample.md:
        1. Process video to get depth maps
        2. Save EXR files with simple naming
        3. Optionally create MP4 visualization
        """
        try:
            print(f"[DEBUG] Starting video processing...")
            
            # Extract parameters
            input_video = job_data["input_video"]  # Should be ProRes MOV from Nuke
            exr_output_dir = job_data["exr_output_dir"]  # depth_exr directory
            first_frame = int(job_data['first_frame'])
            last_frame = int(job_data['last_frame'])
            metric_depth = job_data['metric_depth']
            floating_point = job_data['floating_point']
            encoder = job_data.get('encoder', 'vitl')
            
            # CRITICAL FIX: Proper Windows path handling for input video
            if os.name == 'nt':  # Windows
                if not input_video.startswith('//'):  # Not a UNC path
                    input_video = input_video.replace('/', '\\')
                # Clean up any double backslashes in paths (but preserve UNC \\server\share)
                if not input_video.startswith('\\\\'):
                    input_video = input_video.replace('\\\\', '\\')
            else:
                # Unix-like systems: ensure forward slashes
                input_video = input_video.replace('\\', '/')
            
            # New optimization parameters
            create_source_mp4 = job_data.get('create_source_mp4', False)  # Usually False for optimization
            create_depth_vis_mp4 = job_data.get('create_depth_vis_mp4', True)  # Usually True
            depth_mp4_dir = job_data.get('depth_mp4_dir', os.path.join(os.path.dirname(exr_output_dir), "depth_mp4"))
            
            print(f"[DEBUG] Processing Parameters:")
            print(f"  - Input video: {input_video}")
            print(f"  - EXR output: {exr_output_dir}")
            print(f"  - Create source MP4: {create_source_mp4}")
            print(f"  - Create depth vis MP4: {create_depth_vis_mp4}")
            print(f"  - Depth MP4 dir: {depth_mp4_dir}")
            
            # Default args like sample.md
            max_len = -1
            target_fps = -1
            max_res = -1  # No limit - use original resolution
            input_size = 518
            
            logger.info(f"Processing video: {input_video}")
            logger.info(f"Output directory: {exr_output_dir}")
            logger.info(f"Frame range: {first_frame} to {last_frame}")
            
            # Load model exactly like sample.md
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            checkpoint_name = 'metric_video_depth_anything' if metric_depth else 'video_depth_anything'
            checkpoint_path = job_data[f"{checkpoint_name}_checkpoint"]
            
            # Determine encoder from job data or checkpoint path
            if 'encoder' in job_data:
                encoder = job_data['encoder']
            elif 'vits' in checkpoint_path:
                encoder = 'vits'
            elif 'vitb' in checkpoint_path:
                encoder = 'vitb'
            elif 'vitl' in checkpoint_path:
                encoder = 'vitl'
            else:
                encoder = 'vitl'  # Default to vitl for better quality
            
            logger.info(f"Using encoder: {encoder}")
            model_config = model_configs[encoder]
            
            self.model = VideoDepthAnything(**model_config, metric=metric_depth)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
            self.model = self.model.to(self.device).eval()
            
            print(f"[DEBUG] Model loaded successfully")
            
            # Read frames from ProRes MOV (no EXR conversion needed - already done by Nuke)
            print(f"[LOADING] Reading ProRes MOV from Nuke...")
            
            if input_video.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Video file - use original method
                frames, target_fps = read_video_frames(input_video, max_len, target_fps, max_res)
            else:
                # Fallback for image sequences
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
            
            # Update progress
            if progress_task:
                progress_task.setMessage("Processing depth maps...")
                progress_task.setProgress(25)
            
            # Process using Video-Depth-Anything method
            print(f"[PROCESSING] Using Video-Depth-Anything method...")
            depths, fps = self.model.infer_video_depth(
                frames, 
                target_fps, 
                input_size=input_size, 
                device=self.device, 
                fp32=floating_point == "float32"
            )
            
            logger.info(f"Processing completed: {len(depths)} depth maps generated")
            
            # Apply depth normalization if enabled
            enable_normalization = job_data.get('enable_normalization', False)
            if enable_normalization:
                near_value = job_data.get('near_value', 0.0)
                far_value = job_data.get('far_value', 5.0)
                invert_depth = job_data.get('invert_depth', False)
                
                print(f"[NORMALIZATION] Applying depth normalization:")
                print(f"  - Near value: {near_value}")
                print(f"  - Far value: {far_value}")
                print(f"  - Invert depth: {invert_depth}")
                
                # Apply normalization to each depth frame
                for i, depth in enumerate(depths):
                    depths[i] = self.normalize_depth(depth, near_value, far_value, invert_depth)
                    if i % 10 == 0 or i < 3:
                        print(f"[NORMALIZATION] Processed frame {i+1}/{len(depths)}")
                
                print(f"[NORMALIZATION] Depth normalization completed for {len(depths)} frames")
            else:
                print(f"[NORMALIZATION] Depth normalization disabled - using raw depth values")
            
            # Update progress
            if progress_task:
                progress_task.setMessage("Saving EXR files...")
                progress_task.setProgress(50)
            
            # CRITICAL FIX: Proper Windows path handling
            # Convert forward slashes to backslashes on Windows, but preserve UNC paths
            if os.name == 'nt':  # Windows
                if not exr_output_dir.startswith('//'):  # Not a UNC path
                    exr_output_dir = exr_output_dir.replace('/', '\\')
                # Clean up any double backslashes in paths (but preserve UNC \\server\share)
                if not exr_output_dir.startswith('\\\\'):
                    exr_output_dir = exr_output_dir.replace('\\\\', '\\')
            else:
                # Unix-like systems: ensure forward slashes
                exr_output_dir = exr_output_dir.replace('\\', '/')
            
            # Create output directory
            os.makedirs(exr_output_dir, exist_ok=True)
            print(f"Creating output directory: {exr_output_dir}")
            
            # Verify directory was created
            if not os.path.exists(exr_output_dir):
                raise Exception(f"Failed to create output directory: {exr_output_dir}")
            
            # Save EXR files exactly like sample.md
            # Use EXR frame numbers if provided (for EXR->MOV->EXR workflow), otherwise use input frame range
            exr_first_frame = job_data.get('exr_first_frame', first_frame)
            exr_last_frame = job_data.get('exr_last_frame', last_frame)
            exr_frame_range = range(exr_first_frame, exr_last_frame + 1)
            print(f"[EXR] DEBUG: Created exr_frame_range: {exr_first_frame} to {exr_last_frame} (total: {len(exr_frame_range)} frames)")
            
            # Generate base name from input file (remove frame patterns and extensions)
            import re
            input_base_name = os.path.splitext(os.path.basename(input_video))[0]
            # Remove frame patterns like %04d, ####, etc.
            input_base_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_base_name)
            input_base_name = input_base_name.strip('_.')
            # Add _depth suffix
            depth_base_name = f"{input_base_name}_depth"
            
            print(f"[SAVING] Saving {len(depths)} EXR files...")
            print(f"[DEBUG] Depth data shape: {depths[0].shape if len(depths) > 0 else 'No depths'}")
            print(f"[DEBUG] Input frame range: {first_frame} to {last_frame}")
            print(f"[DEBUG] EXR frame range: {exr_first_frame} to {exr_last_frame}")
            print(f"[DEBUG] Input base name: {input_base_name}")
            print(f"[DEBUG] Depth base name: {depth_base_name}")
            print(f"[DEBUG] Output directory: {exr_output_dir}")
            print(f"[DEBUG] Depths variable type: {type(depths)}")
            print(f"[DEBUG] Depths variable length: {len(depths) if hasattr(depths, '__len__') else 'No length'}")
            
            saved_count = 0
            failed_count = 0
            total_frames = len(depths)
            
            print(f"[EXR] Starting EXR creation for {total_frames} frames...")
            print(f"[EXR] Frame range: {exr_frame_range.start} to {exr_frame_range.stop-1}")
            
            # USE SIMPLE EXR CREATION LIKE SAMPLE.MD - KEEP IT SIMPLE!
            print(f"[EXR] Using simple EXR creation approach (like sample.md)...")
            
            # Get shot name for better organization
            shot_name = os.path.splitext(os.path.basename(input_video))[0]
            # Remove frame pattern tokens like %04d or #### from shot name
            shot_name = re.sub(r'(%0\d+d|#+)', '', shot_name).strip('_.')
            print(f"[EXR] Shot name: {shot_name}")
            
            # Use OpenCV with OpenEXR support (like iMatte does)
            print(f"[EXR] Using OpenCV with OpenEXR support (like iMatte)")
            print(f"[EXR] OpenEXR environment variable is set: OPENCV_IO_ENABLE_OPENEXR=1")
            
            saved_count = 0
            
            # Debug: Check if we have data to process
            print(f"[EXR] DEBUG: exr_frame_range: {list(exr_frame_range)}")
            print(f"[EXR] DEBUG: depths count: {len(depths)}")
            print(f"[EXR] DEBUG: depths type: {type(depths)}")
            if len(depths) > 0:
                print(f"[EXR] DEBUG: first depth shape: {depths[0].shape}")
                print(f"[EXR] DEBUG: first depth type: {type(depths[0])}")
            else:
                print(f"[EXR] ERROR: No depth data available!")
                return {"status": "error", "message": "No depth data available for EXR creation"}
            
            print(f"[EXR] Starting EXR creation loop...")
            
            # Convert depths to list if it's a numpy array
            if hasattr(depths, 'shape') and len(depths.shape) > 1:
                # It's a numpy array - convert to list of individual frames
                depths_list = [depths[i] for i in range(depths.shape[0])]
                print(f"[EXR] DEBUG: Converted numpy array to list of {len(depths_list)} frames")
            else:
                # It's already a list
                depths_list = depths
                print(f"[EXR] DEBUG: Using depths as list of {len(depths_list)} frames")
            
            print(f"[EXR] Starting to save {len(depths_list)} depth frames...")
            
            try:
                for idx, (i, depth) in enumerate(zip(exr_frame_range, depths_list)):
                    # Use shot name with dot format: TML305_021_020_FTG01_proxy_v001_src_depth.1024.exr
                    # CRITICAL: Use os.path.join() to avoid mixed path separators (OpenEXR crashes with mixed slashes!)
                    output_exr = os.path.join(exr_output_dir, f"{shot_name}_depth.{i}.exr")
                    
                    # Show progress every 10 frames
                    if idx % 10 == 0 or idx < 3:
                        print(f"[EXR] Saving frame {idx+1}/{len(depths_list)}: {output_exr}", flush=True)

                    try:
                        # Use OpenCV with OpenEXR support (like iMatte does)
                        # Convert depth to float32 for EXR (values 0-1)
                        exr_depth = depth.astype("float32")
                        
                        # Import cv2 here to ensure OpenEXR environment is set
                        import cv2
                        cv2.imwrite(output_exr, exr_depth)
                        saved_count += 1
                        
                    except cv2.error as cv_err:
                        if "OpenEXR" in str(cv_err):
                            print(f"[EXR] OpenEXR not available, saving as PNG instead")
                            png_path = output_exr.replace('.exr', '.png')
                            # Convert depth to 0-255 range for PNG
                            depth_png = ((depth - depth.min()) / 
                                        (depth.max() - depth.min()) * 255).astype(np.uint8)
                            cv2.imwrite(png_path, depth_png)
                            saved_count += 1
                        else:
                            print(f"[EXR] ERROR: Failed to save EXR file {output_exr}: {str(cv_err)}", flush=True)
                            print(f"[EXR] ERROR: Frame index: {idx}, Frame number: {i}", flush=True)
                            print(f"[EXR] ERROR: Depth shape: {depth.shape}, dtype: {depth.dtype}", flush=True)
                            failed_count += 1
                            continue
                    except Exception as e:
                        print(f"[EXR] ERROR: Failed to save EXR file {output_exr}: {str(e)}", flush=True)
                        print(f"[EXR] ERROR: Frame index: {idx}, Frame number: {i}", flush=True)
                        print(f"[EXR] ERROR: Depth shape: {depth.shape}, dtype: {depth.dtype}", flush=True)
                        import traceback
                        traceback.print_exc()
                        
                        # Try to continue with next frame
                        failed_count += 1
                        continue
                        
            except Exception as e:
                print(f"[EXR] CRITICAL ERROR: EXR creation loop failed at outer level: {str(e)}", flush=True)
                print(f"[EXR] CRITICAL ERROR: This crashed the entire loop!", flush=True)
                import traceback
                print(f"[EXR] Full traceback:", flush=True)
                traceback.print_exc()
                return {"status": "error", "message": f"EXR creation loop failed: {e}"}

            print(f"EXRs successfully written to: {exr_output_dir}")
            print(f"Total EXR files saved: {saved_count}")
            
            # Report EXR saving results
            print(f"[EXR RESULTS] Successfully saved: {saved_count} EXR files")
            failed_count = len(depths) - saved_count
            if failed_count > 0:
                print(f"[EXR RESULTS] Failed to save: {failed_count} EXR files")
                if saved_count == 0:
                    print(f"[ERROR] No EXR files were created! Check the error messages above.")
                    return {"status": "error", "message": f"Failed to create any EXR files. {failed_count} failures."}
                else:
                    print(f"[WARNING] Some EXR files failed, but {saved_count} were saved successfully")
            else:
                print(f"[SUCCESS] All {saved_count} EXR files created successfully!")
            
            print(f"[COMPLETE] EXR creation finished! Files written to: {exr_output_dir}")
            logger.info(f"Processing completed successfully: {exr_output_dir}")
            
            # Update progress
            if progress_task:
                progress_task.setMessage("Creating MP4 files...")
                progress_task.setProgress(75)
            
            # Create MP4 files AFTER EXR creation (if requested)
            if create_depth_vis_mp4:
                print(f"[MP4] Creating MP4 visualization files after EXR creation...")
                try:
                    # Generate input name for MP4 files
                    input_name = os.path.splitext(os.path.basename(input_video))[0]
                    import re
                    input_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_name)
                    input_name = input_name.strip('_.')
                    
                    # Use the provided depth_mp4_dir or create default
                    if not depth_mp4_dir:
                        depth_mp4_dir = os.path.join(os.path.dirname(exr_output_dir), "depth_mp4")
                    
                    # Ensure MP4 directory exists
                    os.makedirs(depth_mp4_dir, exist_ok=True)
                    
                    # Create depth visualization MP4 using the depths data we have
                    if depths is not None and len(depths) > 0:
                        print(f"[MP4] Creating depth visualization MP4 from {len(depths)} depth frames...")
                        
                        # Create depth visualization MP4
                        depth_vis_mp4_path = os.path.join(depth_mp4_dir, f"{input_name}_vis.mp4")
                        
                        # Use the same approach as original run.py with inferno colormap
                        import imageio
                        from matplotlib import cm
                        
                        # Convert depths to numpy array if needed
                        depths_array = np.array(depths)
                        
                        # Use inferno colormap like original save_video function
                        # Handle both old and new matplotlib versions
                        try:
                            # New matplotlib API (3.5+)
                            colormap = np.array(cm.colormaps["inferno"].colors)
                        except AttributeError:
                            # Old matplotlib API (< 3.5)
                            colormap = np.array(cm.get_cmap("inferno").colors)
                        
                        d_min, d_max = depths_array.min(), depths_array.max()
                        
                        print(f"[MP4] Depth range: {d_min:.3f} to {d_max:.3f}")
                        
                        # Create video writer with same settings as original
                        writer = imageio.get_writer(depth_vis_mp4_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
                        
                        for i in range(depths_array.shape[0]):
                            depth = depths_array[i]
                            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
                            writer.append_data(depth_vis)
                            
                            if i % 20 == 0:
                                print(f"[MP4] Processing frame {i+1}/{len(depths)}")
                        
                        writer.close()
                        
                        if os.path.exists(depth_vis_mp4_path):
                            file_size = os.path.getsize(depth_vis_mp4_path) / (1024*1024)  # MB
                            print(f"[MP4] ✅ Depth visualization MP4 created: {depth_vis_mp4_path} ({file_size:.1f} MB)")
                        else:
                            print(f"[MP4] ❌ Failed to create depth visualization MP4")
                    else:
                        print(f"[WARNING] MP4 creation skipped - no depth data available")
                        
                except Exception as e:
                    print(f"[WARNING] MP4 creation failed, but EXR files were saved successfully: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Don't fail the entire process if MP4 creation fails
            else:
                print(f"[MP4] MP4 creation disabled in job settings")
            
            # Update progress
            if progress_task:
                progress_task.setMessage("Completed!")
                progress_task.setProgress(100)
            
            return {
                "status": "success", 
                "output_path": exr_output_dir,
                "depth_frames": len(depths),
                "fps": fps,
                "optimized": True
            }
            
        except Exception as e:
            print(f"[ERROR] Video processing failed: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            logger.error(f"Video processing failed: {str(e)}")
            
            return {"status": "error", "message": str(e)}
    
    
    def _read_exr_frame(self, frame_path):
        """Read a single EXR frame and convert to RGB using OpenCV (like iMatte)"""
        try:
            import cv2
            
            # Read EXR file using OpenCV with OpenEXR support
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                logger.error(f"Failed to read EXR file: {frame_path}")
                return None
            
            # If it's a single channel (grayscale), convert to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # BGR to RGB conversion
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # BGRA to RGB conversion
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Ensure values are in 0-255 range
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to read EXR file {frame_path}: {str(e)}")
            return None

    def normalize_depth(self, depth, near_value, far_value, invert_depth):
        """Normalize depth values for VFX production - Nuke GUI compatibility"""
        try:
            # Convert to numpy if needed
            if not isinstance(depth, np.ndarray):
                depth = np.array(depth)
            
            # Store original range for logging
            original_min = depth.min()
            original_max = depth.max()
            
            # Normalize depth to 0-1 range based on near/far values
            if original_max > original_min:
                # Normalize to 0-1 range first
                depth_normalized = (depth - original_min) / (original_max - original_min)
                
                # Apply near/far mapping
                if far_value > near_value:
                    # Map to near/far range (near=0, far=1)
                    depth_normalized = near_value + (far_value - near_value) * depth_normalized
                else:
                    # Invert the mapping (far=0, near=1)
                    depth_normalized = far_value + (near_value - far_value) * depth_normalized
            else:
                # Constant depth - set to near value
                depth_normalized = np.full_like(depth, near_value)
            
            # Apply inversion if requested
            if invert_depth:
                # Invert the normalized values within the near/far range
                depth_normalized = far_value - (depth_normalized - near_value)
            
            # Log the transformation for debugging
            new_min = depth_normalized.min()
            new_max = depth_normalized.max()
            print(f"[NORMALIZATION] Depth transformed: {original_min:.3f}-{original_max:.3f} -> {new_min:.3f}-{new_max:.3f}")
            
            return depth_normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize depth: {str(e)}")
            print(f"[NORMALIZATION ERROR] Failed to normalize depth: {str(e)}")
            return depth
    
    def _create_mp4_files(self, frames, depths, output_path, input_name, fps, first_frame=1001, last_frame=1100, job_data=None, create_source_mp4=True, create_depth_vis_mp4=True):
        """Create MP4 files with selective creation"""
        try:
            # Create depth_mp4 subdirectory for MP4 files
            depth_mp4_dir = os.path.join(output_path, "depth_mp4")
            os.makedirs(depth_mp4_dir, exist_ok=True)
            
            print(f"[MP4] Creating MP4 files in: {depth_mp4_dir}")
            logger.info(f"Creating MP4 files in: {depth_mp4_dir}")
            
            created_files = []
            
            # Get frame range from job data or use defaults (used for both source and depth videos)
            if job_data:
                mp4_first_frame = job_data.get('exr_first_frame', first_frame)
                mp4_last_frame = job_data.get('exr_last_frame', last_frame)
            else:
                mp4_first_frame = first_frame
                mp4_last_frame = last_frame
            # Use actual frame count instead of calculated range
            mp4_frame_count = len(depths)
            
            print(f"[MP4] Frame range for MP4: {mp4_first_frame}-{mp4_last_frame} ({mp4_frame_count} frames)")
            
            # Save source video ONLY if requested (optimization)
            if create_source_mp4:
                source_video = os.path.join(depth_mp4_dir, f"{input_name}_src.mp4")
                print(f"[MP4] Creating source video: {source_video}")
                
                # Convert frames to numpy array if it's a list
                if isinstance(frames, list):
                    frames = np.array(frames)
                
                # Use processed frames (may be scaled down)
                if len(frames) > 0:
                    current_height, current_width = frames[0].shape[:2]
                    print(f"[MP4] Using processed frame resolution: {current_width}x{current_height}")
                
                # Use same frame range and timecode for source video
                if len(frames) >= mp4_frame_count:
                    mp4_frames = frames[:mp4_frame_count]
                else:
                    mp4_frames = frames
                
                try:
                    print(f"[MP4] Starting source video creation...")
                    # Ensure mp4_frames is a numpy array
                    if isinstance(mp4_frames, list):
                        mp4_frames = np.array(mp4_frames)
                        print(f"[MP4] Converted frame list to numpy array: {mp4_frames.shape}")
                    
                    # Simple MP4 creation without timecode (like sample.md)
                    save_video(mp4_frames, source_video, fps=fps, is_depths=False)
                    print(f"[MP4] Source video created: {source_video}")
                    created_files.append(source_video)
                except Exception as e:
                    print(f"[ERROR] Failed to create source MP4: {str(e)}")
                    print(f"[ERROR] Source video path: {source_video}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[MP4] Skipping source video creation (optimization)")
            
            # Save depth visualization video (always create this)
            if create_depth_vis_mp4:
                # Use shot name for MP4 filename (like sample.md)
                depth_video = os.path.join(depth_mp4_dir, f"{input_name}_vis.mp4")
                print(f"[MP4] Creating depth visualization: {depth_video}")
                
                # Convert depths to numpy array if it's a list
                if isinstance(depths, list):
                    depths = np.array(depths)
                
                # Use only the frames corresponding to the requested frame range
                print(f"[MP4] Total depths available: {len(depths)} frames")
                
                # Slice depths to match the requested frame range
                if len(depths) >= mp4_frame_count:
                    mp4_depths = depths[:mp4_frame_count]  # Use first N frames
                    print(f"[MP4] Using first {mp4_frame_count} frames for MP4")
                else:
                    mp4_depths = depths  # Use all available frames
                    print(f"[MP4] Using all {len(depths)} available frames for MP4")
                
                try:
                    print(f"[MP4] Starting depth video creation...")
                    # Ensure mp4_depths is a numpy array
                    if isinstance(mp4_depths, list):
                        mp4_depths = np.array(mp4_depths)
                        print(f"[MP4] Converted depth list to numpy array: {mp4_depths.shape}")
                    
                    # Simple MP4 creation without timecode (like sample.md)
                    save_video(mp4_depths, depth_video, fps=fps, is_depths=True)
                    print(f"[MP4] Depth visualization created: {depth_video}")
                except Exception as e:
                    print(f"[ERROR] Failed to create depth MP4: {str(e)}")
                    print(f"[ERROR] Depth video path: {depth_video}")
                    import traceback
                    traceback.print_exc()
                
                # EMBED METADATA INTO MP4
                if job_data:
                    try:
                        from metadata_embedder import MetadataEmbedder
                        embedder = MetadataEmbedder()
                        
                        # Create input info from available data
                        input_info = {
                            "resolution": f"{depths.shape[2]}x{depths.shape[1]}",
                            "fps": fps,
                            "shot_name": input_name.replace("_src", ""),
                            "project_name": "Video Depth Anything",
                            "user": os.getenv("USERNAME", "unknown"),
                            "nuke_project": "Nuke Project"
                        }
                        
                        metadata = embedder.create_metadata(job_data, input_info)
                        embedder.embed_mp4_metadata(depth_video, metadata)
                        
                    except Exception as e:
                        print(f"[METADATA] Warning: Could not embed MP4 metadata: {e}")
                
                created_files.append(depth_video)
            else:
                print(f"[MP4] Skipping depth visualization creation")
            
            logger.info(f"MP4 files created successfully: {created_files}")
            
        except Exception as e:
            logger.error(f"Failed to create MP4 files: {str(e)}")
            print(f"[ERROR] Failed to create MP4 files: {str(e)}")
            # Don't re-raise the exception - MP4 creation is optional

def main():
    """Main engine loop - AUTOMATIC OPTIMIZATION DETECTION"""
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
        print("VDA ENGINE - COMMAND LINE MODE")
        print("=" * 60)
        print(f"Processing job file: {job_file}")
        print("=" * 60)
        
        try:
            print(f"[DEBUG] Loading job file: {job_file}")
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            print(f"[DEBUG] Job data loaded successfully")
            
            # AUTOMATIC OPTIMIZATION DETECTION
            is_optimized = (
                job_data.get('create_source_mp4') is False and 
                job_data.get('create_depth_vis_mp4') is True and
                'depth_mp4_dir' in job_data
            )
            
            if is_optimized:
                print("=" * 60)
                print("VIDEO PROCESSING STARTED")
                print("=" * 60)
            else:
                print("=" * 60)
                print("ORIGINAL WORKFLOW")
                print("=" * 60)
            
            print(f"[DEBUG] Creating engine instance...")
            engine = OriginalVideoDepthEngine()
            print(f"[DEBUG] Engine created successfully")
            
            print(f"[DEBUG] Starting video processing...")
            if is_optimized:
                result = engine.process_video_optimized(job_data)
                print(f"[DEBUG] Video processing completed")
            else:
                result = engine.process_video_original(job_data)
                print(f"[DEBUG] ORIGINAL video processing completed")
            
            print("=" * 60)
            print("PROCESSING COMPLETED!")
            print("=" * 60)
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Output: {result.get('output_path', 'Unknown')}")
                print(f"Frames processed: {result.get('depth_frames', 0)}")
                print(f"FPS: {result.get('fps', 24)}")
                if result.get('optimized'):
                    print("+ Video processing completed")
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
            print("=" * 60)
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
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
                    
                    # AUTOMATIC OPTIMIZATION DETECTION for job queue mode
                    is_optimized = (
                        job_data.get('create_source_mp4') is False and 
                        job_data.get('create_depth_vis_mp4') is True and
                        'depth_mp4_dir' in job_data
                    )
                    
                    if is_optimized:
                        print("=" * 60)
                        print("VIDEO PROCESSING STARTED")
                        print("=" * 60)
                        result = engine.process_video_optimized(job_data)
                    else:
                        print("=" * 60)
                        print("ORIGINAL WORKFLOW")
                        print("=" * 60)
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