#!/usr/bin/env python3
"""
Metadata Embedder for Video Depth Anything
Embeds metadata into MOV, EXR, and MP4 files
Based on sample.md patterns but enhanced for production use
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MetadataEmbedder:
    """Embeds metadata into various file formats"""
    
    def __init__(self):
        self.creation_time = datetime.now().isoformat()
    
    def create_metadata(self, job_data, input_info=None):
        """
        Create comprehensive metadata dictionary
        
        Args:
            job_data: Job configuration data
            input_info: Additional input information
            
        Returns:
            dict: Metadata dictionary
        """
        metadata = {
            # Basic Info
            "creation_time": self.creation_time,
            "software": "Video Depth Anything",
            "version": "1.0",
            "workflow": "optimized",
            
            # Input Information
            "input_source": job_data.get("input_video", ""),
            "input_frame_range": f"{job_data.get('first_frame', 1)}-{job_data.get('last_frame', 1)}",
            "input_resolution": input_info.get("resolution", "unknown") if input_info else "unknown",
            "input_fps": input_info.get("fps", "unknown") if input_info else "unknown",
            
            # Processing Information
            "model_encoder": job_data.get("encoder", "vits"),
            "model_checkpoint": job_data.get("video_depth_anything_checkpoint", ""),
            "floating_point": job_data.get("floating_point", "float32"),
            "metric_depth": job_data.get("metric_depth", False),
            
            # Depth Processing
            "depth_normalization": {
                "enabled": job_data.get("enable_normalization", False),
                "near_value": job_data.get("near_value", 0.0),
                "far_value": job_data.get("far_value", 5.0),
                "invert_depth": job_data.get("invert_depth", False)
            },
            
            # Output Information
            "output_frame_range": f"{job_data.get('exr_first_frame', 1001)}-{job_data.get('exr_last_frame', 1001)}",
            "output_format": "EXR",
            "output_precision": "32-bit float",
            
            # Technical Details
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_utc": datetime.utcnow().isoformat(),
            "file_format_version": "1.0",
            
            # Workflow Information
            "workflow_type": "EXR->MOV->Depth",
            "optimization_enabled": True,
            "source_mp4_created": job_data.get("create_source_mp4", False),
            "depth_vis_created": job_data.get("create_depth_vis_mp4", True),
        }
        
        # Add custom metadata if provided
        if input_info:
            if "shot_name" in input_info:
                metadata["shot_name"] = input_info["shot_name"]
            if "project_name" in input_info:
                metadata["project_name"] = input_info["project_name"]
            if "user" in input_info:
                metadata["processed_by"] = input_info["user"]
            if "nuke_project" in input_info:
                metadata["nuke_project"] = input_info["nuke_project"]
        
        return metadata
    
    def embed_exr_metadata(self, exr_path, metadata):
        """
        Embed metadata into EXR file using OpenEXR
        
        Args:
            exr_path: Path to EXR file
            metadata: Metadata dictionary
        """
        try:
            import OpenEXR
            import Imath
            
            # Read existing EXR file
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            
            # Add metadata to header
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # Convert nested dicts to JSON strings
                    header[key] = json.dumps(value)
                else:
                    header[key] = str(value)
            
            # Read pixel data
            channels = header["channels"]
            pixel_data = {}
            for channel in channels:
                pixel_data[channel] = exr_file.channel(channel)
            
            exr_file.close()
            
            # Write new EXR with metadata
            new_exr_file = OpenEXR.OutputFile(exr_path, header)
            new_exr_file.writePixels(pixel_data)
            new_exr_file.close()
            
            logger.info(f"Metadata embedded in EXR: {exr_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed EXR metadata: {e}")
            return False
    
    def embed_mov_metadata(self, mov_path, metadata):
        """
        Embed metadata into MOV file using ffmpeg
        
        Args:
            mov_path: Path to MOV file
            metadata: Metadata dictionary
        """
        try:
            import subprocess
            
            # Convert metadata to ffmpeg format
            metadata_args = []
            for key, value in metadata.items():
                if isinstance(value, dict):
                    value = json.dumps(value)
                metadata_args.extend(["-metadata", f"{key}={value}"])
            
            # Create temporary file for output
            temp_path = mov_path.replace('.mov', '_temp.mov')
            
            # Use ffmpeg to add metadata
            cmd = [
                "ffmpeg", "-i", mov_path,
                *metadata_args,
                "-c", "copy",  # Copy without re-encoding
                "-y",  # Overwrite output file
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with metadata-enhanced version
                os.replace(temp_path, mov_path)
                logger.info(f"Metadata embedded in MOV: {mov_path}")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
                
        except Exception as e:
            logger.error(f"Failed to embed MOV metadata: {e}")
            return False
    
    def embed_mp4_metadata(self, mp4_path, metadata):
        """
        Embed metadata into MP4 file using ffmpeg
        
        Args:
            mp4_path: Path to MP4 file
            metadata: Metadata dictionary
        """
        try:
            import subprocess
            
            # Convert metadata to ffmpeg format
            metadata_args = []
            for key, value in metadata.items():
                if isinstance(value, dict):
                    value = json.dumps(value)
                metadata_args.extend(["-metadata", f"{key}={value}"])
            
            # Create temporary file for output
            temp_path = mp4_path.replace('.mp4', '_temp.mp4')
            
            # Use ffmpeg to add metadata
            cmd = [
                "ffmpeg", "-i", mp4_path,
                *metadata_args,
                "-c", "copy",  # Copy without re-encoding
                "-y",  # Overwrite output file
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with metadata-enhanced version
                os.replace(temp_path, mp4_path)
                logger.info(f"Metadata embedded in MP4: {mp4_path}")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
                
        except Exception as e:
            logger.error(f"Failed to embed MP4 metadata: {e}")
            return False
    
    def create_metadata_file(self, output_dir, metadata, filename="metadata.json"):
        """
        Create a separate metadata JSON file
        
        Args:
            output_dir: Output directory
            metadata: Metadata dictionary
            filename: Metadata filename
        """
        try:
            metadata_path = os.path.join(output_dir, filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata file created: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to create metadata file: {e}")
            return None

def test_metadata_embedding():
    """Test the metadata embedding functionality"""
    
    # Sample job data
    job_data = {
        "input_video": "test_input.mov",
        "first_frame": 1,
        "last_frame": 10,
        "encoder": "vits",
        "video_depth_anything_checkpoint": "models/video_depth_anything_vits.pth",
        "floating_point": "float32",
        "metric_depth": False,
        "enable_normalization": True,
        "near_value": 0.0,
        "far_value": 5.0,
        "invert_depth": False,
        "exr_first_frame": 1001,
        "exr_last_frame": 1010,
        "create_source_mp4": False,
        "create_depth_vis_mp4": True
    }
    
    # Sample input info
    input_info = {
        "resolution": "3840x2160",
        "fps": 24.0,
        "shot_name": "TML304_012_030_FTG01",
        "project_name": "TML Project",
        "user": "wasim",
        "nuke_project": "TML_comp_v001.nk"
    }
    
    # Create metadata embedder
    embedder = MetadataEmbedder()
    
    # Generate metadata
    metadata = embedder.create_metadata(job_data, input_info)
    
    print("Generated Metadata:")
    print(json.dumps(metadata, indent=2))
    
    # Test metadata file creation
    test_dir = "test_metadata_output"
    os.makedirs(test_dir, exist_ok=True)
    
    metadata_file = embedder.create_metadata_file(test_dir, metadata)
    
    if metadata_file:
        print(f"\nMetadata file created: {metadata_file}")
    
    return metadata

if __name__ == "__main__":
    test_metadata_embedding()
