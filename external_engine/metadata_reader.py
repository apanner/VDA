#!/usr/bin/env python3
"""
Metadata Reader for Video Depth Anything
Reads embedded metadata from EXR, MOV, and MP4 files
"""

import os
import json
import subprocess
from pathlib import Path

def read_exr_metadata(exr_path):
    """
    Read metadata from EXR file
    
    Args:
        exr_path: Path to EXR file
        
    Returns:
        dict: Metadata dictionary
    """
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        
        metadata = {}
        for key, value in header.items():
            if key not in ["channels", "dataWindow", "displayWindow", "lineOrder", "pixelAspectRatio", "screenWindowCenter", "screenWindowWidth"]:
                try:
                    # Try to parse as JSON if it looks like JSON
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        metadata[key] = json.loads(value)
                    else:
                        metadata[key] = value
                except:
                    metadata[key] = str(value)
        
        exr_file.close()
        return metadata
        
    except Exception as e:
        print(f"Error reading EXR metadata: {e}")
        return {}

def read_mov_metadata(mov_path):
    """
    Read metadata from MOV file using ffprobe
    
    Args:
        mov_path: Path to MOV file
        
    Returns:
        dict: Metadata dictionary
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", mov_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            metadata = {}
            
            # Extract format metadata
            if "format" in data and "tags" in data["format"]:
                metadata.update(data["format"]["tags"])
            
            # Extract stream metadata
            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                if "tags" in stream:
                    metadata.update(stream["tags"])
            
            return metadata
        else:
            print(f"ffprobe failed: {result.stderr}")
            return {}
            
    except Exception as e:
        print(f"Error reading MOV metadata: {e}")
        return {}

def read_mp4_metadata(mp4_path):
    """
    Read metadata from MP4 file using ffprobe
    
    Args:
        mp4_path: Path to MP4 file
        
    Returns:
        dict: Metadata dictionary
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", mp4_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            metadata = {}
            
            # Extract format metadata
            if "format" in data and "tags" in data["format"]:
                metadata.update(data["format"]["tags"])
            
            # Extract stream metadata
            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                if "tags" in stream:
                    metadata.update(stream["tags"])
            
            return metadata
        else:
            print(f"ffprobe failed: {result.stderr}")
            return {}
            
    except Exception as e:
        print(f"Error reading MP4 metadata: {e}")
        return {}

def test_metadata_reading():
    """Test reading metadata from generated files"""
    
    print("=" * 70)
    print("TESTING METADATA READING")
    print("=" * 70)
    
    # Test EXR metadata reading
    exr_dir = "test_metadata_output/depth_exr"
    if os.path.exists(exr_dir):
        exr_files = [f for f in os.listdir(exr_dir) if f.endswith('.exr')]
        if exr_files:
            exr_path = os.path.join(exr_dir, exr_files[0])
            print(f"\n[EXR] Reading metadata from: {exr_path}")
            
            exr_metadata = read_exr_metadata(exr_path)
            if exr_metadata:
                print("EXR Metadata found:")
                for key, value in exr_metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("No metadata found in EXR file")
    
    # Test MP4 metadata reading
    mp4_dir = "test_metadata_output/depth_mp4"
    if os.path.exists(mp4_dir):
        mp4_files = [f for f in os.listdir(mp4_dir) if f.endswith('.mp4')]
        if mp4_files:
            mp4_path = os.path.join(mp4_dir, mp4_files[0])
            print(f"\n[MP4] Reading metadata from: {mp4_path}")
            
            mp4_metadata = read_mp4_metadata(mp4_path)
            if mp4_metadata:
                print("MP4 Metadata found:")
                for key, value in mp4_metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("No metadata found in MP4 file")
    
    # Test metadata.json file
    metadata_file = "test_metadata_output/metadata.json"
    if os.path.exists(metadata_file):
        print(f"\n[JSON] Reading metadata from: {metadata_file}")
        try:
            with open(metadata_file, 'r') as f:
                json_metadata = json.load(f)
            print("JSON Metadata found:")
            for key, value in json_metadata.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error reading JSON metadata: {e}")

if __name__ == "__main__":
    test_metadata_reading()
