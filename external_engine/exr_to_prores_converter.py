#!/usr/bin/env python3
"""
EXR to Apple ProRes Converter
Converts EXR image sequences to high-quality Apple ProRes .mov files
for Video Depth Anything processing
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
from pathlib import Path
import logging

try:
    import OpenEXR
    import Imath
    EXR_AVAILABLE = True
except ImportError:
    EXR_AVAILABLE = False
    print("WARNING: OpenEXR not available. EXR conversion will not work.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not available. EXR conversion will not work.")

logger = logging.getLogger(__name__)

class EXRToProResConverter:
    """Convert EXR image sequences to Apple ProRes .mov files"""
    
    def __init__(self):
        self.temp_dir = None
        self.temp_prores_file = None
        
    def convert_exr_sequence_to_prores(self, exr_sequence_path, first_frame, last_frame, fps=24.0, quality="high"):
        """
        Convert EXR image sequence to Apple ProRes .mov file
        
        Args:
            exr_sequence_path: Path to EXR sequence (e.g., "sequence.####.exr")
            first_frame: First frame number
            last_frame: Last frame number
            fps: Frame rate for output video
            quality: ProRes quality ("high", "medium", "low")
            
        Returns:
            str: Path to temporary ProRes .mov file
        """
        if not EXR_AVAILABLE:
            raise RuntimeError("OpenEXR not available - cannot convert EXR sequences")
        
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available - cannot convert EXR sequences")
        
        print(f"[EXR->ProRes] Converting EXR sequence to Apple ProRes...")
        print(f"   Input: {exr_sequence_path}")
        print(f"   Frames: {first_frame}-{last_frame}")
        print(f"   FPS: {fps}")
        print(f"   Quality: {quality}")
        
        # Create temporary directory for conversion
        self.temp_dir = tempfile.mkdtemp(prefix="exr_prores_")
        print(f"   Temp directory: {self.temp_dir}")
        
        # Convert EXR frames to temporary images
        temp_images = self._convert_exr_frames_to_images(
            exr_sequence_path, first_frame, last_frame
        )
        
        if not temp_images:
            raise RuntimeError("No EXR frames were successfully converted")
        
        # Create ProRes video from images
        self.temp_prores_file = self._create_prores_video(
            temp_images, fps, quality
        )
        
        # Clean up temporary images
        self._cleanup_temp_images(temp_images)
        
        print(f"[EXR->ProRes] Conversion completed: {self.temp_prores_file}")
        return self.temp_prores_file
    
    def _convert_exr_frames_to_images(self, exr_sequence_path, first_frame, last_frame):
        """Convert EXR frames to temporary image files"""
        temp_images = []
        
        print(f"[EXR->ProRes] Converting {last_frame - first_frame + 1} EXR frames...")
        
        for frame_num in range(first_frame, last_frame + 1):
            # Generate frame path
            frame_path = exr_sequence_path.replace("####", f"{frame_num:04d}")
            frame_path = frame_path.replace("%04d", f"{frame_num:04d}")
            
            if not os.path.exists(frame_path):
                print(f"   [WARNING] Frame {frame_num} not found: {frame_path}")
                continue
            
            try:
                # Read EXR frame
                rgb_image = self._read_exr_to_rgb(frame_path)
                
                if rgb_image is not None:
                    # Save as temporary image
                    temp_image_path = os.path.join(
                        self.temp_dir, f"frame_{frame_num:04d}.png"
                    )
                    cv2.imwrite(temp_image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    temp_images.append(temp_image_path)
                    print(f"   [OK] Frame {frame_num} converted")
                else:
                    print(f"   [ERROR] Failed to read frame {frame_num}")
                    
            except Exception as e:
                print(f"   [ERROR] Failed to convert frame {frame_num}: {str(e)}")
                continue
        
        print(f"[EXR->ProRes] Successfully converted {len(temp_images)} frames")
        return temp_images
    
    def _read_exr_to_rgb(self, exr_path):
        """Read EXR file and convert to RGB numpy array"""
        try:
            # Open EXR file
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            
            # Get image dimensions
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Read RGB channels
            channels = ['R', 'G', 'B']
            channel_data = {}
            
            for channel in channels:
                if channel in header['channels']:
                    # Read channel data
                    channel_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                    channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                else:
                    # If channel doesn't exist, create zeros
                    channel_data[channel] = np.zeros(width * height, dtype=np.float32)
            
            # Reshape and stack channels
            for channel in channels:
                channel_data[channel] = channel_data[channel].reshape((height, width))
            
            # Stack channels to create RGB image
            img = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=2)
            
            # Handle different EXR formats and color spaces
            # ACEScg to sRGB conversion (simplified)
            img = self._acescg_to_srgb(img)
            
            # Clamp values to 0-1 range and convert to 0-255
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            
            exr_file.close()
            return img
            
        except Exception as e:
            logger.error(f"Failed to read EXR file {exr_path}: {str(e)}")
            return None
    
    def _acescg_to_srgb(self, img):
        """Convert ACEScg color space to sRGB (simplified)"""
        try:
            # Simple ACEScg to sRGB conversion
            # This is a basic approximation - for production use, implement proper color management
            
            # ACEScg has a different white point and gamma
            # For now, we'll use a simple linear to sRGB conversion
            img = np.clip(img, 0, None)  # Remove negative values
            
            # Simple gamma correction (2.2)
            img = np.power(img, 1.0/2.2)
            
            return img
            
        except Exception as e:
            logger.warning(f"Color space conversion failed, using original: {str(e)}")
            return img
    
    def _create_prores_video(self, image_paths, fps, quality):
        """Create Apple ProRes video from image sequence using FFmpeg"""
        try:
            # Sort image paths by frame number
            image_paths.sort()
            
            # Create ProRes output path
            prores_output = os.path.join(self.temp_dir, "sequence_prores.mov")
            
            # FFmpeg command for Apple ProRes
            # Using ProRes 422 HQ for high quality (more compatible)
            prores_preset = {
                "high": "prores",         # ProRes 422 HQ (most compatible)
                "medium": "prores",       # ProRes 422 HQ
                "low": "prores_lt"        # ProRes 422 LT
            }.get(quality, "prores")
            
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate", str(fps),
                "-i", os.path.join(self.temp_dir, "frame_%04d.png"),
                "-c:v", prores_preset,
                "-pix_fmt", "yuv422p10le",  # 10-bit ProRes
                "-r", str(fps),
                prores_output
            ]
            
            print(f"[EXR->ProRes] Creating ProRes video with FFmpeg...")
            print(f"   Command: {' '.join(cmd)}")
            
            # Run FFmpeg
            print(f"[EXR->ProRes] Running FFmpeg command...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"[EXR->ProRes] FFmpeg return code: {result.returncode}")
            if result.stdout:
                print(f"[EXR->ProRes] FFmpeg STDOUT: {result.stdout}")
            if result.stderr:
                print(f"[EXR->ProRes] FFmpeg STDERR: {result.stderr}")
            
            if result.returncode != 0:
                print(f"[ERROR] FFmpeg failed:")
                print(f"   STDOUT: {result.stdout}")
                print(f"   STDERR: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Check if output file was created
            if not os.path.exists(prores_output):
                raise RuntimeError("ProRes output file was not created")
            
            file_size = os.path.getsize(prores_output) / (1024 * 1024)  # MB
            print(f"[EXR->ProRes] ProRes video created: {file_size:.1f} MB")
            
            return prores_output
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg conversion timed out")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found - please install FFmpeg")
        except Exception as e:
            raise RuntimeError(f"Failed to create ProRes video: {str(e)}")
    
    def _cleanup_temp_images(self, temp_images):
        """Clean up temporary image files"""
        for img_path in temp_images:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp image {img_path}: {str(e)}")
    
    def cleanup(self):
        """Clean up all temporary files"""
        if self.temp_prores_file and os.path.exists(self.temp_prores_file):
            try:
                os.remove(self.temp_prores_file)
                print(f"[EXR->ProRes] Cleaned up ProRes file: {self.temp_prores_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up ProRes file: {str(e)}")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f"[EXR->ProRes] Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {str(e)}")
        
        self.temp_prores_file = None
        self.temp_dir = None

def is_exr_sequence(input_path):
    """Check if input path is an EXR image sequence"""
    if not input_path:
        return False
    
    # Must contain .exr extension
    if '.exr' not in input_path.lower():
        return False
    
    # Check for EXR sequence patterns
    exr_patterns = ['####', '%04d']
    return any(pattern in input_path for pattern in exr_patterns)

def convert_exr_to_prores_for_depth(input_path, first_frame, last_frame, fps=24.0):
    """
    Main function to convert EXR sequence to ProRes for depth processing
    
    Args:
        input_path: EXR sequence path
        first_frame: First frame number
        last_frame: Last frame number
        fps: Frame rate
        
    Returns:
        str: Path to temporary ProRes file, or None if conversion failed
    """
    if not is_exr_sequence(input_path):
        return None
    
    converter = EXRToProResConverter()
    try:
        prores_path = converter.convert_exr_sequence_to_prores(
            input_path, first_frame, last_frame, fps, quality="high"
        )
        return prores_path
    except Exception as e:
        print(f"[ERROR] EXR to ProRes conversion failed: {str(e)}")
        converter.cleanup()
        return None

if __name__ == "__main__":
    # Test the converter
    print("EXR to Apple ProRes Converter Test")
    print("=" * 50)
    
    # Example usage
    test_exr_path = "test_sequence.####.exr"
    if os.path.exists(test_exr_path.replace("####", "0001")):
        prores_path = convert_exr_to_prores_for_depth(test_exr_path, 1, 3, 24.0)
        if prores_path:
            print(f"Test successful: {prores_path}")
        else:
            print("Test failed")
    else:
        print("No test EXR sequence found")
