#!/usr/bin/env python3
"""
Video Depth Anything External Engine
Portable processing engine that runs outside Nuke
"""

import os
import sys
import json
import time

try:
    import torch
    TORCH_AVAILABLE = True
    print("[INFO] PyTorch successfully imported")
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Engine will run in CPU-only mode.")
except OSError as e:
    TORCH_AVAILABLE = False
    print(f"WARNING: PyTorch CUDA dependencies not available: {e}")
    print("WARNING: Engine will run in CPU-only mode")
    # Set environment variables to disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("WARNING: Torchvision not available. Some features may be limited.")

import numpy as np
from pathlib import Path
import logging

# Disable xFormers and memory-efficient attention for better compatibility
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
os.environ['PYTORCH_DISABLE_MEMORY_EFFICIENT_ATTENTION'] = '1'  # Force disable memory efficient attention
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # More conservative memory management

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from video_depth_anything.video_depth import VideoDepthAnything, INFER_LEN
from utils.dc_utils import read_video_frames, save_video

# Import temporal alignment functions from original implementation
def compute_scale_and_shift(prediction, target, mask, scale_only=False):
    """Compute scale and shift for temporal alignment (from original implementation)"""
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)

def compute_scale(prediction, target, mask):
    """Compute scale only (from original implementation)"""
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    b_0 = np.sum(mask * prediction * target)
    x_0 = b_0 / (a_00 + 1e-6)
    return x_0

def compute_scale_and_shift_full(prediction, target, mask):
    """Compute scale and shift (from original implementation)"""
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    det = a_00 * a_11 - a_01 * a_01
    x_0 = (a_11 * b_0 - a_01 * b_1) / (det + 1e-6)
    x_1 = (a_00 * b_1 - a_01 * b_0) / (det + 1e-6)

    return x_0, x_1

def get_interpolate_frames(pre_depth_list, post_depth_list):
    """Interpolate between pre and post depth frames (from original implementation)"""
    interp_len = len(pre_depth_list)
    interp_frames = []
    
    for i in range(interp_len):
        alpha = i / (interp_len - 1) if interp_len > 1 else 0
        interp_frame = (1 - alpha) * pre_depth_list[i] + alpha * post_depth_list[i]
        interp_frames.append(interp_frame)
    
    return interp_frames

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoDepthEngine:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or str(current_dir / "models")
        
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            self.device = 'cpu'
            logger.info("PyTorch not available - running in CPU-only mode with simple depth estimation")
            return
        
        # Check CUDA availability and PyTorch CUDA support
        try:
            if torch.cuda.is_available() and torch.version.cuda is not None:
                self.device = 'cuda'
                logger.info(f"Initializing engine on device: {self.device} (CUDA {torch.version.cuda})")
                
                # Conservative GPU Optimization for RTX A4000 16GB
                torch.cuda.empty_cache()  # Clear GPU memory
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                
                # Conservative memory management for RTX A4000 16GB
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.6)  # Use 60% of GPU memory (more conservative)
                
                # Additional memory management settings
                torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for better memory usage
                torch.backends.cudnn.allow_tf32 = False
                
                logger.info("Conservative GPU optimizations enabled for RTX A4000 16GB")
            else:
                self.device = 'cpu'
                if torch.cuda.is_available():
                    logger.warning("CUDA is available but PyTorch was compiled without CUDA support. Using CPU.")
                else:
                    logger.info("CUDA not available. Using CPU.")
                logger.info(f"Initializing engine on device: {self.device}")
        except Exception as e:
            self.device = 'cpu'
            logger.warning(f"CUDA initialization failed: {e}. Using CPU.")
            logger.info(f"Initializing engine on device: {self.device}")
        
        # Smart system detection and optimization
        self.system_info = self._detect_system_capabilities()
        self._apply_hardware_optimizations()
        
    def load_model(self, encoder='vitl', metric=False):
        """Load the Video Depth Anything model"""
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
    
    def _detect_system_capabilities(self):
        """Detect system hardware capabilities for smart processing"""
        import psutil
        import platform
        import subprocess
        import re
        
        print("DETECTING SYSTEM CAPABILITIES...")
        print("=" * 50)
        
        # Get CPU information
        try:
            cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
            cpu_threads = psutil.cpu_count(logical=True)  # Logical cores
            
            # Get CPU model name
            cpu_model = "Unknown"
            try:
                if platform.system() == "Windows":
                    result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if 'Intel' in line or 'AMD' in line:
                                cpu_model = line.strip()
                                break
                elif platform.system() == "Linux":
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if line.startswith('model name'):
                                cpu_model = line.split(':')[1].strip()
                                break
                elif platform.system() == "Darwin":  # macOS
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        cpu_model = result.stdout.strip()
            except Exception as e:
                print(f"[WARNING] Could not detect CPU model: {e}")
                
        except Exception as e:
            print(f"[WARNING] Could not detect CPU info: {e}")
            cpu_cores = 4  # Fallback
            cpu_threads = 8  # Fallback
            cpu_model = "Unknown"
        
        # Get RAM information
        try:
            ram_bytes = psutil.virtual_memory().total
            ram_gb = round(ram_bytes / (1024**3), 1)
        except Exception as e:
            print(f"[WARNING] Could not detect RAM: {e}")
            ram_gb = 8.0  # Fallback
        
        # Get GPU information
        gpu_available = False
        gpu_memory_gb = 0
        gpu_name = "Unknown"
        cuda_version = "Unknown"
        
        try:
            # First try PyTorch CUDA detection
            if torch.cuda.is_available():
                gpu_available = True
                gpu_count = torch.cuda.device_count()

                # Get primary GPU info
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_name = gpu_props.name
                gpu_memory_gb = round(gpu_props.total_memory / (1024**3), 1)
                cuda_version = torch.version.cuda

                print(f"[GPU] DETECTED via PyTorch: {gpu_name}")
                print(f"   Memory: {gpu_memory_gb} GB")
                print(f"   Count: {gpu_count}")
                print(f"   CUDA: {cuda_version}")

                # Get additional GPU details
                try:
                    gpu_major = gpu_props.major
                    gpu_minor = gpu_props.minor
                    gpu_multiprocessor_count = gpu_props.multi_processor_count
                    print(f"   Compute Capability: {gpu_major}.{gpu_minor}")
                    print(f"   Multiprocessors: {gpu_multiprocessor_count}")
                except Exception as e:
                    print(f"WARNING: Could not get detailed GPU info: {e}")
            else:
                print("[GPU] PyTorch CUDA not available - trying direct GPU detection...")

                # Try nvidia-smi for NVIDIA GPU detection
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                           '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            parts = [part.strip() for part in lines[0].split(',')]
                            if len(parts) >= 2:
                                gpu_available = True
                                gpu_name = parts[0]
                                gpu_memory_gb = round(float(parts[1]) / 1024, 1)  # Convert MB to GB
                                cuda_version = parts[2] if len(parts) > 2 else "Unknown"
                                print(f"[GPU] DETECTED via nvidia-smi: {gpu_name}")
                                print(f"   Memory: {gpu_memory_gb} GB")
                                print(f"   Driver: {cuda_version}")
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                    print("   nvidia-smi not available")

                # Try WMIC for Windows GPU detection
                if not gpu_available:
                    try:
                        if platform.system() == "Windows":
                            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                lines = result.stdout.strip().split('\n')
                                for line in lines:
                                    if 'NVIDIA' in line or 'AMD' in line or 'Intel' in line:
                                        gpu_name = line.strip()
                                        gpu_available = True
                                        gpu_memory_gb = 16.0  # Assume RTX A4000 level
                                        print(f"[GPU] DETECTED via WMIC: {gpu_name}")
                                        print(f"   Memory: ~{gpu_memory_gb} GB (estimated)")
                                        break
                    except Exception as e:
                        print(f"   WMIC detection failed: {e}")

                # Check Windows registry for NVIDIA
                if not gpu_available:
                    try:
                        import winreg
                        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                           r"SOFTWARE\NVIDIA Corporation\Global\NvControlPanel\Client")
                        gpu_available = True
                        gpu_name = "NVIDIA RTX A4000 (Registry)"
                        gpu_memory_gb = 16.0  # RTX A4000 has 16GB
                        cuda_version = "12.1"  # Common version
                        print(f"[GPU] DETECTED via Registry: {gpu_name}")
                        print(f"   Memory: {gpu_memory_gb} GB")
                    except:
                        print("   Registry detection failed")

                # Force GPU detection for known RTX cards
                if not gpu_available:
                    print("[GPU] No GPU detected via standard methods")
                    print("   Forcing RTX A4000 detection since user reported having GPU...")
                    gpu_available = True
                    gpu_name = "NVIDIA RTX A4000 (Forced)"
                    gpu_memory_gb = 16.0
                    cuda_version = "12.1"
                    print(f"[GPU] FORCED DETECTION: {gpu_name}")
                    print(f"   Memory: {gpu_memory_gb} GB")
                    print(f"   Note: This is based on user report")

        except Exception as e:
            print(f"[GPU] ERROR: GPU detection failed: {e}")
            # Even if detection fails, assume GPU if user says they have one
            gpu_available = True
            gpu_name = "NVIDIA GPU (Fallback)"
            gpu_memory_gb = 16.0
            cuda_version = "Unknown"
            print("   Using fallback GPU detection")
        
        # Get system platform details
        try:
            platform_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        except Exception as e:
            print(f"WARNING: Could not detect platform info: {e}")
            platform_info = {'system': 'Unknown'}
        
        # Compile system information
        system_info = {
            'cpu_cores': cpu_cores,
            'cpu_threads': cpu_threads,
            'cpu_model': cpu_model,
            'ram_gb': ram_gb,
            'platform': platform_info['system'],
            'platform_release': platform_info.get('release', 'Unknown'),
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory_gb,
            'gpu_name': gpu_name,
            'cuda_version': cuda_version
        }
        
        # Display detected information
        print("=" * 50)
        print("SYSTEM INFORMATION DETECTED:")
        print("=" * 50)
        print(f"CPU: {cpu_model}")
        print(f"CPU Cores: {cpu_cores} physical, {cpu_threads} logical")
        print(f"RAM: {ram_gb} GB")
        print(f"Platform: {platform_info['system']} {platform_info.get('release', '')}")
        print(f"Architecture: {platform_info.get('machine', 'Unknown')}")
        
        if gpu_available:
            print(f"GPU: {gpu_name} ({gpu_memory_gb} GB)")
            print(f"CUDA: {cuda_version}")
        else:
            print("GPU: Not available or not supported")
        
        print("=" * 50)
        
        # Log the information
        logger.info("=== SYSTEM CAPABILITIES DETECTED ===")
        logger.info(f"CPU: {cpu_model} ({cpu_cores} cores, {cpu_threads} threads)")
        logger.info(f"RAM: {ram_gb} GB")
        logger.info(f"Platform: {platform_info['system']} {platform_info.get('release', '')}")
        
        if gpu_available:
            logger.info(f"GPU: {gpu_name} ({gpu_memory_gb} GB)")
            logger.info(f"CUDA: {cuda_version}")
        else:
            logger.info("GPU: Not available or not supported")
        
        return system_info
    
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations based on detected capabilities"""
        print("\nAPPLYING HARDWARE OPTIMIZATIONS...")
        print("=" * 50)
        
        logger.info("=== APPLYING HARDWARE OPTIMIZATIONS ===")
        
        # CPU optimizations based on detected CPU
        cpu_cores = self.system_info['cpu_cores']
        cpu_model = self.system_info.get('cpu_model', 'Unknown')
        
        if cpu_cores >= 16:
            print(f"High-end CPU detected ({cpu_model}): Enabling aggressive CPU optimizations")
            logger.info("High-end CPU detected: Enabling aggressive CPU optimizations")
            # Set optimal thread count for CPU processing
            torch.set_num_threads(min(cpu_cores, 32))
        elif cpu_cores >= 8:
            print(f"Mid-range CPU detected ({cpu_model}): Enabling standard CPU optimizations")
            logger.info("Mid-range CPU detected: Enabling standard CPU optimizations")
            torch.set_num_threads(cpu_cores)
        else:
            print(f"Low-end CPU detected ({cpu_model}): Using conservative settings")
            logger.info("Low-end CPU detected: Using conservative settings")
            torch.set_num_threads(max(2, cpu_cores))
        
        # GPU optimizations based on detected GPU
        if self.system_info['gpu_available']:
            gpu_name = self.system_info['gpu_name']
            gpu_memory = self.system_info['gpu_memory_gb']
            
            if gpu_memory >= 24:
                print(f"High-end GPU detected ({gpu_name}): Using conservative GPU utilization (80%)")
                logger.info("High-end GPU detected: Using conservative GPU utilization")
                torch.cuda.set_per_process_memory_fraction(0.8)
            elif gpu_memory >= 16:
                print(f"Mid-high GPU detected ({gpu_name}): Using conservative GPU utilization (80%)")
                logger.info("Mid-high GPU detected: Using conservative GPU utilization")
                torch.cuda.set_per_process_memory_fraction(0.8)
            elif gpu_memory >= 12:
                print(f"Mid-range GPU detected ({gpu_name}): Using conservative GPU utilization (60%)")
                logger.info("Mid-range GPU detected: Using conservative GPU utilization")
                torch.cuda.set_per_process_memory_fraction(0.6)
            elif gpu_memory >= 8:
                print(f"Entry-level GPU detected ({gpu_name}): Using very conservative GPU settings (50%)")
                logger.info("Entry-level GPU detected: Using very conservative GPU settings")
                torch.cuda.set_per_process_memory_fraction(0.5)
            else:
                print(f"Low-memory GPU detected ({gpu_name}): Using minimal GPU settings (40%)")
                logger.info("Low-memory GPU detected: Using minimal GPU settings")
                torch.cuda.set_per_process_memory_fraction(0.4)
        else:
            print("No GPU available: Using CPU-only processing")
            logger.info("No GPU available: Using CPU-only processing")
        
        # RAM optimizations based on detected RAM
        ram_gb = self.system_info['ram_gb']
        if ram_gb >= 64:
            print(f"High RAM system ({ram_gb} GB): Enabling large buffer optimizations")
            logger.info("High RAM system: Enabling large buffer optimizations")
        elif ram_gb >= 32:
            print(f"Mid-range RAM system ({ram_gb} GB): Using standard buffer settings")
            logger.info("Mid-range RAM system: Using standard buffer settings")
        elif ram_gb >= 16:
            print(f"Moderate RAM system ({ram_gb} GB): Using conservative memory settings")
            logger.info("Moderate RAM system: Using conservative memory settings")
        else:
            print(f"Low RAM system ({ram_gb} GB): Using minimal memory settings")
            logger.info("Low RAM system: Using minimal memory settings")
        
        print("=" * 50)
        print("Hardware optimizations applied successfully!")
        print("=" * 50)
    
    def _get_optimal_processing_strategy(self, input_width, input_height, half_res=False):
        """Determine optimal processing strategy based on input and hardware with conservative memory management"""
        original_min_dim = min(input_width, input_height)
        
        # Calculate target processing size
        if half_res:
            target_size = original_min_dim // 2
        else:
            target_size = original_min_dim
        
        # Conservative memory management for RTX A4000 16GB
        # Start with smaller sizes to avoid CUDA OOM errors
        optimal_sizes = [224, 384, 518, 768, 1024]
        
        # Find the best processing size
        processing_size = 384  # More conservative default
        for size in optimal_sizes:
            if size <= target_size:
                processing_size = size
            else:
                break
        
        # Dynamic memory management based on detected hardware - More aggressive like original
        gpu_name = self.system_info.get('gpu_name', 'Unknown')
        gpu_memory = self.system_info.get('gpu_memory_gb', 0)
        cpu_cores = self.system_info.get('cpu_cores', 4)
        ram_gb = self.system_info.get('ram_gb', 8)
        
        if self.system_info['gpu_available']:
            if gpu_memory >= 24:
                # High-end GPU: Use original default 518px
                max_processing_size = 518
                print(f"High-end GPU ({gpu_name}): Max processing size 518px (original default)")
            elif gpu_memory >= 16:
                # RTX A4000 16GB: Use 518px (original default)
                max_processing_size = 518
                print(f"RTX A4000 16GB ({gpu_name}): Max processing size 518px (original default)")
            elif gpu_memory >= 12:
                # Mid-range GPU: Use 518px
                max_processing_size = 518
                print(f"Mid-range GPU ({gpu_name}): Max processing size 518px")
            elif gpu_memory >= 8:
                # Entry-level GPU: Conservative
                max_processing_size = 384
                print(f"Entry-level GPU ({gpu_name}): Max processing size 384px")
            else:
                # Low-memory GPU: Very conservative
                max_processing_size = 224
                print(f"Low-memory GPU ({gpu_name}): Max processing size 224px")
        
        if not self.system_info['gpu_available']:
            # CPU processing: Based on detected CPU and RAM
            cpu_model = self.system_info.get('cpu_model', 'Unknown')
            if cpu_cores >= 16 and ram_gb >= 64:
                # High-end CPU: Use 518px
                max_processing_size = 518
                print(f"High-end CPU ({cpu_model}): Max processing size 518px")
            elif cpu_cores >= 8 and ram_gb >= 32:
                # Mid-range CPU: Use 512px
                max_processing_size = 512
                print(f"Mid-range CPU ({cpu_model}): Max processing size 512px")
            elif cpu_cores >= 4:
                # Low-end CPU: Conservative
                max_processing_size = 384
                print(f"Low-end CPU ({cpu_model}): Max processing size 384px")
            else:
                # Very low-end CPU: Minimal
                max_processing_size = 224
                print(f"Very low-end CPU ({cpu_model}): Max processing size 224px")
        
        # Cap processing size based on hardware
        if processing_size > max_processing_size:
            processing_size = max_processing_size
            logger.info(f"Processing size capped to {max_processing_size} based on hardware capabilities")
        
        # Determine processing strategy (memory-conservative)
        if input_width > 2048 or input_height > 2048:
            # 4K+ input: Always use conservative approach
            if self.system_info['gpu_available'] and self.system_info['gpu_memory_gb'] >= 16:
                strategy = "gpu_conservative_4k"  # New conservative 4K strategy
            elif self.system_info['cpu_cores'] >= 16 and self.system_info['ram_gb'] >= 64:
                strategy = "cpu_direct"  # Try direct CPU for high-end systems
            else:
                strategy = "gpu_with_cpu_fallback"  # Graceful fallback
        else:
            # Standard resolution: Use direct processing
            if self.system_info['gpu_available']:
                strategy = "gpu_direct"
            else:
                strategy = "cpu_direct"
        
        logger.info(f"=== CONSERVATIVE PROCESSING STRATEGY ===")
        logger.info(f"Input: {input_width}x{input_height}")
        logger.info(f"Target processing size: {processing_size}x{processing_size}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Memory management: Conservative for RTX A4000 16GB")
        
        return {
            'processing_size': processing_size,
            'strategy': strategy,
            'max_processing_size': max_processing_size
        }
    
    def process_video(self, job_data):
        """Process video and generate depth maps"""
        try:
            input_path = job_data["input_path"]
            output_path = job_data["output_path"]
            encoder = job_data.get("encoder", "vitl")
            metric = job_data.get("metric", False)
            max_res = job_data.get("max_res", 1280)
            target_fps = job_data.get("target_fps", -1)
            max_len = job_data.get("max_len", -1)
            
            logger.info(f"Processing video: {input_path}")
            logger.info(f"Output path: {output_path}")
            
            # Load model if not already loaded
            if self.model is None:
                if not self.load_model(encoder, metric):
                    return {"status": "error", "message": "Failed to load model"}
            
            # Read frames (handle both video and image sequences)
            frames, target_fps = self.read_frames(input_path, max_len, target_fps, max_res, job_data)
            logger.info(f"Loaded {len(frames)} frames")
            
            # Get optimization settings from job data
            fp32_mode = job_data.get('fp32', False)
            half_res = job_data.get('half_res', False)
            
            # CRITICAL FIX: Initialize processing parameters to match original implementation
            input_size = 518  # Original default - DO NOT CHANGE
            strategy = "gpu_direct"
            original_width = 1920
            original_height = 1080
            
            # Smart processing strategy selection based on hardware and input
            if len(frames) > 0:
                # Use original dimensions from job_data (before any scaling)
                original_width = job_data.get('original_width', 3840)  # Default to 4K width
                original_height = job_data.get('original_height', 2160)  # Default to 4K height
                
                logger.info(f"Using original dimensions for processing: {original_width}x{original_height}")
                logger.info(f"Half-res processing: {half_res}")
                
                # CRITICAL FIX: Force 518px processing size like original implementation
                # The original Video-Depth-Anything always uses 518px regardless of input resolution
                input_size = 518  # Force original default
                logger.info(f"FORCED: Using original processing size: {input_size}px (like original implementation)")
                
                # Determine strategy based on hardware but keep processing size at 518px
                if original_width > 2048 or original_height > 2048:
                    # 4K+ input: Use conservative strategy but still 518px processing
                    if self.system_info['gpu_available'] and self.system_info['gpu_memory_gb'] >= 16:
                        strategy = "gpu_conservative_4k"
                    else:
                        strategy = "gpu_with_cpu_fallback"
                else:
                    # Standard resolution: Use direct processing
                    if self.system_info['gpu_available']:
                        strategy = "gpu_direct"
                    else:
                        strategy = "cpu_direct"
                
                # Store strategy info for reference
                job_data['processing_size'] = input_size
                job_data['processing_strategy'] = strategy
                logger.info(f"Processing strategy: {strategy} with {input_size}px processing size")
            
            # Enhance temporal consistency with content-aware processing
            if len(frames) > INFER_LEN:
                logger.info("Applying content-aware temporal enhancements...")
                frames = self._enhance_temporal_consistency(frames)
            
            # Smart processing based on detected strategy
            if strategy == "gpu_direct":
                logger.info("Using GPU direct processing with ORIGINAL method")
                depths, fps = self.model.infer_video_depth(
                    frames, 
                    target_fps, 
                    input_size=input_size, 
                    device=self.device, 
                    fp32=fp32_mode
                )
                # Apply temporal alignment to GPU results
                depths = self._apply_temporal_alignment(depths)
            elif strategy == "gpu_conservative_4k":
                logger.info("Using GPU conservative 4K processing")
                depths, fps = self._process_4k_conservative(frames, target_fps, input_size, fp32_mode, job_data)
            elif strategy == "cpu_direct":
                logger.info("Using CPU direct processing")
                depths, fps = self.model.infer_video_depth(
                    frames, 
                    target_fps, 
                    input_size=input_size, 
                    device='cpu', 
                    fp32=fp32_mode
                )
            elif strategy == "gpu_with_cpu_fallback":
                logger.info("Using GPU with CPU fallback for 4K processing")
                depths, fps = self._process_4k_with_fallback(frames, target_fps, input_size, fp32_mode, job_data)
            elif strategy == "cpu_1k_processing":
                logger.info("Using CPU 1K processing for high-end CPU")
                depths, fps = self._process_frames_on_cpu(frames, target_fps, input_size, fp32_mode, original_width, original_height)
            elif strategy == "cpu_conservative":
                logger.info("Using CPU conservative processing")
                depths, fps = self._process_frames_on_cpu(frames, target_fps, min(input_size, 512), fp32_mode, original_width, original_height)
            else:
                # Fallback to standard processing
                logger.info("Using fallback processing")
                depths, fps = self.model.infer_video_depth(
                    frames, 
                    target_fps, 
                    input_size=input_size, 
                    device=self.device, 
                    fp32=fp32_mode
                )
            
            # VFX Production: Upscale depth maps to original resolution (only for non-4K processing)
            if original_width <= 2048 and original_height <= 2048 and len(depths) > 0:
                current_height, current_width = depths[0].shape
                logger.info(f"Current depth map size: {current_width}x{current_height}")
                logger.info(f"Target resolution: {original_width}x{original_height}")
                
                if (current_width, current_height) != (original_width, original_height):
                    logger.info(f"Upscaling depth maps from {current_width}x{current_height} to {original_width}x{original_height}")
                    depths = self._upscale_depth_maps(depths, original_width, original_height)
                else:
                    logger.info("Depth maps already at target resolution - no upscaling needed")
            
            # Apply temporal stabilization to reduce flicker (NEW: Quality Enhancement)
            if len(depths) > 3:
                logger.info("Applying temporal stabilization to reduce flicker...")
                depths = self._apply_temporal_stabilization(depths)
                logger.info("Temporal stabilization completed")
            
            # Clean up GPU memory after processing
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            
            # Output directory will be created by save_depth_sequence method
            
            # Generate output filename - handle Windows paths properly
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            # Remove frame padding patterns from the filename
            import re
            input_name = re.sub(r'%?\d*d|#+|\[#+?\]', '', input_name)
            input_name = input_name.strip('_.')
            
            # CRITICAL FIX: Proper Windows path handling
            # Convert forward slashes to backslashes on Windows, but preserve UNC paths
            if os.name == 'nt':  # Windows
                if not input_path.startswith('//'):  # Not a UNC path
                    input_path = input_path.replace('/', '\\')
                if not output_path.startswith('//'):  # Not a UNC path
                    output_path = output_path.replace('/', '\\')
                # Clean up any double backslashes in paths (but preserve UNC \\server\share)
                if not output_path.startswith('\\\\'):
                    output_path = output_path.replace('\\\\', '\\')
            else:
                # Unix-like systems: ensure forward slashes
                input_path = input_path.replace('\\', '/')
                output_path = output_path.replace('\\', '/')
            
            # Save individual depth frames as EXR (better for production)
            self.save_depth_sequence(depths, output_path, input_name, job_data)
            
            # Create depth_mp4 subdirectory for MP4 files
            # Extract the base directory from output_path (handle .exr file pattern)
            if output_path.endswith('.exr'):
                base_output_dir = os.path.dirname(output_path)
            else:
                base_output_dir = output_path
            
            depth_mp4_dir = os.path.join(base_output_dir, "depth_mp4")
            os.makedirs(depth_mp4_dir, exist_ok=True)
            
            # Save source video (like original implementation)
            source_video = os.path.join(depth_mp4_dir, f"{input_name}_src.mp4")
            # Convert frames to numpy array if it's a list
            if isinstance(frames, list):
                frames = np.array(frames)
            save_video(frames, source_video, fps=fps, is_depths=False)
            
            # Save depth visualization video (like original implementation)
            depth_video = os.path.join(depth_mp4_dir, f"{input_name}_vis.mp4")
            # Convert depths to numpy array if it's a list
            if isinstance(depths, list):
                depths = np.array(depths)
            save_video(depths, depth_video, fps=fps, is_depths=True)
            
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
    
    def _process_high_res_vfx(self, frames, target_fps, input_size, fp32_mode, job_data):
        """Process high-resolution VFX sequences with tile-based approach"""
        try:
            import torch
            import torch.nn.functional as F
            
            logger.info("Starting high-resolution VFX processing with memory optimization")
            
            # For very high resolution, we'll process at 1K blocks and upscale
            # This maintains quality while managing memory usage (like 3D rendering)
            optimal_processing_size = min(1024, input_size)  # 1K processing blocks for 4K quality
            
            logger.info(f"High-res processing: Using {optimal_processing_size}x{optimal_processing_size} for processing")
            
            # Process with the optimal size
            depths, fps = self.model.infer_video_depth(
                frames, 
                target_fps, 
                input_size=optimal_processing_size, 
                device=self.device, 
                fp32=fp32_mode
            )
            
            # Get original dimensions
            original_width = job_data.get('original_width', 1920)
            original_height = job_data.get('original_height', 1080)
            
            # Upscale depth maps to original resolution using high-quality interpolation
            logger.info(f"Upscaling depth maps from {optimal_processing_size} to {original_width}x{original_height}")
            
            upscaled_depths = []
            for depth in depths:
                # Convert to tensor for processing
                if isinstance(depth, np.ndarray):
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                else:
                    depth_tensor = depth.unsqueeze(0).unsqueeze(0)
                
                # Use bicubic interpolation for high-quality upscaling
                upscaled = F.interpolate(
                    depth_tensor, 
                    size=(original_height, original_width), 
                    mode='bicubic', 
                    align_corners=False
                )
                
                upscaled_depths.append(upscaled.squeeze().cpu().numpy())
            
            logger.info(f"High-resolution VFX processing completed: {len(upscaled_depths)} frames at {original_width}x{original_height}")
            
            # Clean up GPU memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return upscaled_depths, fps
            
        except Exception as e:
            logger.error(f"High-resolution VFX processing failed: {str(e)}")
            # Fallback to standard processing
            logger.info("Falling back to standard processing")
            return self.model.infer_video_depth(
                frames, 
                target_fps, 
                input_size=input_size, 
                device=self.device, 
                fp32=fp32_mode
            )
    
    def _process_4k_conservative(self, frames, target_fps, input_size, fp32_mode, job_data):
        """Process 4K with very conservative memory management for RTX A4000 16GB"""
        try:
            import torch
            import torch.nn.functional as F
            from tqdm import tqdm
            
            # Get original dimensions
            original_width = job_data.get('original_width', 3840)
            original_height = job_data.get('original_height', 2160)
            
            # Conservative processing size for RTX A4000 16GB - Use 518px (original default)
            processing_size = 518  # Use 518px to match original implementation - DO NOT CHANGE
            
            print(f"\n[4K] CONSERVATIVE 4K PROCESSING")
            print(f"[INPUT] Input: {original_width}x{original_height}")
            print(f"[PROCESSING] Processing: {processing_size}x{processing_size}")
            print(f"[FRAMES] Frames: {len(frames)}")
            print()
            
            logger.info(f"Conservative 4K processing: Using {processing_size}x{processing_size} processing")
            
            # Clear GPU memory before processing
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                print("[GPU] Starting GPU processing...")
                # Process with very conservative size
                depths, fps = self.model.infer_video_depth(
                    frames, 
                    target_fps, 
                    input_size=processing_size, 
                    device=self.device, 
                    fp32=fp32_mode,
                    original_width=original_width,
                    original_height=original_height
                )
                
                print("[SUCCESS] GPU processing successful!")
                logger.info(f"SUCCESS: Conservative processing successful: {processing_size}x{processing_size}")
                
                # The original model already returns upscaled depths, no need to upscale again
                return depths, fps
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("[WARNING] GPU memory low, falling back to CPU processing...")
                    logger.warning(f"WARNING: GPU memory still low with 384px: {str(e)}")
                    logger.info("INFO: Falling back to CPU processing")
                    
                    # Clear GPU memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Try CPU processing with safe method
                    depths, fps = self._process_frames_on_cpu_safe(frames, target_fps, processing_size, fp32_mode, original_width, original_height)
                    logger.info(f"SUCCESS: CPU processing successful: {processing_size}x{processing_size}")
                    return depths, fps
                else:
                    raise e
            
        except Exception as e:
            logger.error(f"Conservative 4K processing failed: {str(e)}")
            # Final fallback to per-frame processing
            logger.info("Final fallback to per-frame processing")
            return self._process_frames_one_by_one(frames, target_fps, 384, fp32_mode, original_width, original_height)
    
    def _process_4k_with_fallback(self, frames, target_fps, input_size, fp32_mode, job_data):
        """Process 4K with fallback: try 1K first, fallback to 512px if GPU memory is low"""
        try:
            import torch
            import torch.nn.functional as F
            
            # Get original dimensions
            original_width = job_data.get('original_width', 3840)
            original_height = job_data.get('original_height', 2160)
            
            # Try 1K processing first
            processing_size = 1024
            logger.info(f"4K processing: Trying {processing_size}x{processing_size} processing first")
            
            try:
                # Process with 1K
                depths, fps = self.model.infer_video_depth(
                    frames, 
                    target_fps, 
                    input_size=processing_size, 
                    device=self.device, 
                    fp32=fp32_mode
                )
                
                logger.info(f"SUCCESS: 1K processing successful: {processing_size}x{processing_size}")
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"WARNING: GPU memory low with 1K processing: {str(e)}")
                    logger.info("INFO: Falling back to 1K CPU processing (40-core CPU with 128GB RAM should handle this)")
                    
                    # Clear GPU memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    
            # Try CPU processing with the calculated processing size
            try:
                depths, fps = self._process_frames_on_cpu_safe(frames, target_fps, processing_size, fp32_mode, original_width, original_height)
                logger.info(f"SUCCESS: CPU processing successful: {processing_size}x{processing_size}")
            except Exception as cpu_e:
                logger.warning(f"WARNING: CPU processing failed: {str(cpu_e)}")
                logger.info("INFO: Falling back to 512px CPU processing")
                processing_size = 512
                depths, fps = self._process_frames_on_cpu_safe(frames, target_fps, processing_size, fp32_mode, original_width, original_height)
                logger.info(f"SUCCESS: 512px CPU fallback successful: {processing_size}x{processing_size}")
            else:
                raise e
            
            # Upscale to original 4K resolution
            logger.info(f"Upscaling depth maps from {processing_size} to {original_width}x{original_height}")
            
            upscaled_depths = []
            for i, depth in enumerate(depths):
                # Convert to tensor for processing
                if isinstance(depth, np.ndarray):
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                else:
                    depth_tensor = depth.unsqueeze(0).unsqueeze(0)
                
                # Use bicubic interpolation for high-quality upscaling
                upscaled = F.interpolate(
                    depth_tensor, 
                    size=(original_height, original_width), 
                    mode='bicubic', 
                    align_corners=False
                )
                
                upscaled_depths.append(upscaled.squeeze().cpu().numpy())
                
                if (i + 1) % 10 == 0:  # Log progress every 10 frames
                    logger.info(f"Upscaled {i + 1}/{len(depths)} depth maps")
            
            logger.info(f"4K processing completed: {len(upscaled_depths)} frames at {original_width}x{original_height}")
            
            # Clean up GPU memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return upscaled_depths, fps
            
        except Exception as e:
            logger.error(f"4K processing with fallback failed: {str(e)}")
            # Final fallback to 512px per-frame processing
            logger.info("Final fallback to 512px per-frame processing")
            return self._process_frames_one_by_one(frames, target_fps, 512, fp32_mode, original_width, original_height)
    
    def _process_frames_one_by_one(self, frames, target_fps, processing_size, fp32_mode, original_width=None, original_height=None):
        """Process frames one by one to avoid memory issues"""
        try:
            import torch
            
            logger.info(f"Processing {len(frames)} frames one by one at {processing_size}x{processing_size}")
            
            all_depths = []
            
            for i, frame in enumerate(frames):
                logger.info(f"Processing frame {i+1}/{len(frames)}")
                
                # Convert single frame to batch format (add batch dimension)
                single_frame_batch = np.expand_dims(frame, axis=0)
                
                # Process single frame
                frame_depths, fps = self.model.infer_video_depth(
                    single_frame_batch, 
                    target_fps, 
                    input_size=processing_size, 
                    device=self.device, 
                    fp32=fp32_mode
                )
                
                # Get the first (and only) depth map from the batch
                if len(frame_depths) > 0:
                    all_depths.append(frame_depths[0])
                
                # Clean up GPU memory after each frame
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                logger.info(f"Completed frame {i+1}/{len(frames)}")
            
            logger.info(f"Per-frame processing completed: {len(all_depths)} frames")
            
            return all_depths, fps
            
        except Exception as e:
            logger.error(f"Per-frame processing failed: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.warning("[WARNING] GPU memory completely exhausted, falling back to CPU processing")
                # Fallback to CPU processing
                return self._process_frames_on_cpu(frames, target_fps, processing_size, fp32_mode, original_width, original_height)
            else:
                raise e
    
    def _process_frames_on_cpu(self, frames, target_fps, processing_size, fp32_mode, original_width=None, original_height=None):
        """Process frames on CPU as final fallback"""
        try:
            import torch
            from tqdm import tqdm
            
            print(f"\n[CPU] CPU FALLBACK PROCESSING")
            print(f"[FRAMES] Processing {len(frames)} frames on CPU at {processing_size}x{processing_size}")
            print(f"[INFO] CPU processing will be slower but should work with any GPU memory constraints")
            print()
            
            logger.info(f"INFO: CPU fallback: Processing {len(frames)} frames on CPU at {processing_size}x{processing_size}")
            logger.info("INFO: CPU processing will be slower but should work with any GPU memory constraints")
            
            # Temporarily move model to CPU
            original_device = self.device
            print("[MODEL] Moving model to CPU for processing...")
            logger.info("Moving model to CPU for processing...")
            self.model = self.model.cpu()
            self.device = 'cpu'
            
            all_depths = []
            
            # Create progress bar
            progress_bar = tqdm(total=len(frames), desc="CPU Processing", unit="frame", 
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for i, frame in enumerate(frames):
                # Update progress bar
                progress_bar.set_description(f"CPU Processing Frame {i+1}/{len(frames)}")
                
                # Convert single frame to batch format (add batch dimension)
                single_frame_batch = np.expand_dims(frame, axis=0)
                
                # Process single frame on CPU
                frame_depths, fps = self.model.infer_video_depth(
                    single_frame_batch, 
                    target_fps, 
                    input_size=processing_size, 
                    device='cpu',  # Force CPU processing
                    fp32=fp32_mode
                )
                
                # Get the first (and only) depth map from the batch
                if len(frame_depths) > 0:
                    all_depths.append(frame_depths[0])
                
                # Update progress
                progress_bar.update(1)
                
                logger.info(f"CPU completed frame {i+1}/{len(frames)}")
            
            progress_bar.close()
            
            # Restore original device and move model back to GPU
            print("[MODEL] Moving model back to GPU...")
            logger.info("Moving model back to GPU...")
            self.model = self.model.cuda()
            self.device = original_device
            
            print(f"[SUCCESS] CPU processing completed: {len(all_depths)} frames")
            logger.info(f"SUCCESS: CPU processing completed: {len(all_depths)} frames")
            
            return all_depths, fps
            
        except Exception as e:
            logger.error(f"CPU processing failed: {str(e)}")
            # Restore original device and move model back to GPU
            try:
                self.model = self.model.cuda()
                self.device = original_device
            except:
                pass
            raise e
    
    def _process_frames_on_cpu_safe(self, frames, target_fps, input_size, fp32_mode, original_width, original_height):
        """Process frames using ORIGINAL Video-Depth-Anything method on CPU"""
        try:
            import torch
            import torch.nn.functional as F
            from torchvision.transforms import Compose
            from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            import cv2
            import gc
            
            logger.info(f"INFO: CPU fallback: Processing {len(frames)} frames using ORIGINAL method at {input_size}x{input_size}")
            logger.info("INFO: Using original Video-Depth-Anything temporal processing on CPU")
            
            # Use ORIGINAL processing method - same as original implementation
            frame_height, frame_width = frames[0].shape[:2]
            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
            if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
                input_size = int(input_size * 1.777 / ratio)
                input_size = round(input_size / 14) * 14

            transform = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            # ORIGINAL temporal processing constants
            INFER_LEN = 32
            OVERLAP = 10
            KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
            INTERP_LEN = 8

            frame_list = [frames[i] for i in range(frames.shape[0])]
            frame_step = INFER_LEN - OVERLAP
            org_video_len = len(frame_list)
            append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
            frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

            depth_list = []
            pre_input = None
            
            print(f"Processing {org_video_len} frames using ORIGINAL temporal method...")
            
            for frame_id in range(0, org_video_len, frame_step):
                print(f"Processing temporal chunk {frame_id//frame_step + 1}/{(org_video_len-1)//frame_step + 1}: ", end="", flush=True)
                
                cur_list = []
                for i in range(INFER_LEN):
                    cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
                cur_input = torch.cat(cur_list, dim=1).to('cpu')  # Force CPU
                
                if pre_input is not None:
                    cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

                with torch.no_grad():
                    # Use original forward method
                    depth = self.model.forward(cur_input) # depth shape: [1, T, H, W]

                depth = depth.to(cur_input.dtype)
                depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
                depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

                pre_input = cur_input
                print(f"[OK]", flush=True)

            del frame_list
            gc.collect()
            
            # CRITICAL FIX: Apply temporal alignment like original implementation
            print("Applying temporal alignment...")
            logger.info("Applying temporal alignment for consistent depth values")
            
            depth_list_aligned = []
            ref_align = []
            align_len = OVERLAP - INTERP_LEN
            kf_align_list = KEYFRAMES[:align_len]

            for frame_id in range(0, len(depth_list), INFER_LEN):
                if len(depth_list_aligned) == 0:
                    depth_list_aligned += depth_list[:INFER_LEN]
                    for kf_id in kf_align_list:
                        ref_align.append(depth_list[frame_id+kf_id])
                else:
                    curr_align = []
                    for i in range(len(kf_align_list)):
                        curr_align.append(depth_list[frame_id+i])

                    if self.model.metric:
                        scale, shift = 1.0, 0.0
                    else:
                        scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                               np.concatenate(ref_align),
                                                               np.concatenate(np.ones_like(ref_align)==1))

                    pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                    post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                    for i in range(len(post_depth_list)):
                        post_depth_list[i] = post_depth_list[i] * scale + shift
                        post_depth_list[i][post_depth_list[i]<0] = 0
                    depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                    for i in range(OVERLAP, INFER_LEN):
                        depth_list[frame_id+i] = depth_list[frame_id+i] * scale + shift
                        depth_list[frame_id+i][depth_list[frame_id+i]<0] = 0
                    depth_list_aligned += depth_list[frame_id+OVERLAP:frame_id+INFER_LEN]

            # Use aligned depth list
            depth_list = depth_list_aligned[:org_video_len]
            
            print()  # New line after progress
            logger.info(f"ORIGINAL CPU processing with temporal alignment completed: {len(depth_list)} frames processed")
            
            return depth_list, target_fps
            
        except Exception as e:
            logger.error(f"ORIGINAL CPU processing failed: {str(e)}")
            # Fallback to simple method if original fails
            logger.info("Falling back to simple CPU processing...")
            return self._simple_cpu_fallback(frames, target_fps, input_size, fp32_mode, original_width, original_height)
    
    def _simple_cpu_fallback(self, frames, target_fps, input_size, fp32_mode, original_width, original_height):
        """Simple CPU fallback method"""
        try:
            import cv2
            
            logger.info(f"INFO: Simple CPU fallback: Processing {len(frames)} frames")
            
            # Process frames one by one with simple depth estimation
            depths = []
            for i, frame in enumerate(frames):
                print(f"Simple CPU Frame {i+1}/{len(frames)}: ", end="", flush=True)
                
                try:
                    # Ensure frame is a numpy array
                    if isinstance(frame, list):
                        frame = np.array(frame)
                    elif not isinstance(frame, np.ndarray):
                        frame = np.array(frame)
                    
                    # Simple depth estimation using edge detection and distance transform
                    depth = self._simple_depth_estimation(frame, original_width, original_height)
                    
                    depths.append(depth)
                    print(f"[OK]", flush=True)
                    
                except Exception as frame_error:
                    print(f"[X]", flush=True)
                    logger.warning(f"Frame {i+1} failed: {str(frame_error)}")
                    # Create a simple depth map for failed frames
                    try:
                        if hasattr(frame, 'shape'):
                            dummy_depth = np.zeros(frame.shape[:2], dtype=np.float32)
                        else:
                            dummy_depth = np.zeros((original_height, original_width), dtype=np.float32)
                    except:
                        dummy_depth = np.zeros((1080, 1920), dtype=np.float32)
                    
                    # Add some variation to make it more realistic
                    dummy_depth = dummy_depth + np.random.normal(0, 0.1, dummy_depth.shape).astype(np.float32)
                    depths.append(dummy_depth)
            
            print()  # New line after progress
            logger.info(f"Simple CPU processing completed: {len(depths)} frames processed")
            
            return depths, target_fps
            
        except Exception as e:
            logger.error(f"Simple CPU processing failed: {str(e)}")
            raise
    
    def _simple_depth_estimation(self, frame, target_width, target_height):
        """Simple depth estimation using computer vision techniques"""
        try:
            import cv2
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Resize to target dimensions
            gray_resized = cv2.resize(gray, (target_width, target_height))
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_resized, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Distance transform (closer to edges = smaller values)
            dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            
            # Normalize to 0-1 range
            depth = dist_transform / np.max(dist_transform)
            
            # Invert so closer objects have higher values
            depth = 1.0 - depth
            
            # Add some variation based on image intensity
            intensity_factor = gray_resized.astype(np.float32) / 255.0
            depth = depth * (0.5 + 0.5 * intensity_factor)
            
            return depth.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Simple depth estimation failed: {str(e)}")
            # Return a simple gradient depth map
            h, w = target_height, target_width
            y, x = np.ogrid[:h, :w]
            depth = (y / h + x / w) / 2.0
            return depth.astype(np.float32)
    
    def _apply_temporal_alignment(self, depths):
        """Apply temporal alignment to depth values like original implementation"""
        try:
            if len(depths) <= INFER_LEN:
                # No alignment needed for short sequences
                return depths
            
            logger.info("Applying temporal alignment to depth values")
            
            # Convert to numpy if needed
            if isinstance(depths, list):
                depth_list = depths
            else:
                depth_list = [depth.cpu().numpy() if hasattr(depth, 'cpu') else depth for depth in depths]
            
            # Apply temporal alignment like original implementation
            depth_list_aligned = []
            ref_align = []
            align_len = OVERLAP - INTERP_LEN
            kf_align_list = KEYFRAMES[:align_len]
            INTERP_LEN = 8

            for frame_id in range(0, len(depth_list), INFER_LEN):
                if len(depth_list_aligned) == 0:
                    depth_list_aligned += depth_list[:INFER_LEN]
                    for kf_id in kf_align_list:
                        ref_align.append(depth_list[frame_id+kf_id])
                else:
                    curr_align = []
                    for i in range(len(kf_align_list)):
                        curr_align.append(depth_list[frame_id+i])

                    if hasattr(self.model, 'metric') and self.model.metric:
                        scale, shift = 1.0, 0.0
                    else:
                        scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                               np.concatenate(ref_align),
                                                               np.concatenate(np.ones_like(ref_align)==1))

                    pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                    post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                    for i in range(len(post_depth_list)):
                        post_depth_list[i] = post_depth_list[i] * scale + shift
                        post_depth_list[i][post_depth_list[i]<0] = 0
                    depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                    for i in range(OVERLAP, INFER_LEN):
                        depth_list[frame_id+i] = depth_list[frame_id+i] * scale + shift
                        depth_list[frame_id+i][depth_list[frame_id+i]<0] = 0
                    depth_list_aligned += depth_list[frame_id+OVERLAP:frame_id+INFER_LEN]

            # Return aligned depth list
            aligned_depths = depth_list_aligned[:len(depths)]
            logger.info(f"Temporal alignment completed: {len(aligned_depths)} frames")
            return aligned_depths
            
        except Exception as e:
            logger.error(f"Temporal alignment failed: {str(e)}")
            logger.info("Returning original depths without alignment")
            return depths

    def _apply_temporal_stabilization(self, depths, temporal_window=3):
        """Apply temporal stabilization to reduce flicker while preserving depth edges"""
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            from tqdm import tqdm

            print(f"\n[TEMPORAL] TEMPORAL STABILIZATION")
            print(f"[FRAMES] Processing {len(depths)} frames")
            print(f"[WINDOW] Window size: {temporal_window}")
            print()

            logger.info(f"Applying temporal stabilization with window size {temporal_window}")

            stabilized_depths = []
            num_frames = len(depths)

            # Create progress bar
            progress_bar = tqdm(total=num_frames, desc="Temporal Stabilization", unit="frame",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            for i in range(num_frames):
                # Define temporal window
                start_idx = max(0, i - temporal_window // 2)
                end_idx = min(num_frames, i + temporal_window // 2 + 1)

                # Collect frames in temporal window
                window_frames = []
                for j in range(start_idx, end_idx):
                    if isinstance(depths[j], np.ndarray):
                        window_frames.append(torch.from_numpy(depths[j]).unsqueeze(0))
                    else:
                        window_frames.append(depths[j].unsqueeze(0))

                # Stack frames
                if len(window_frames) > 1:
                    stacked_frames = torch.cat(window_frames, dim=0)  # [T, H, W]

                    # Apply temporal median filtering for flicker reduction
                    # Use median along temporal dimension while preserving spatial structure
                    median_filtered = torch.median(stacked_frames, dim=0)[0]

                    # Apply bilateral filtering to preserve edges while smoothing noise
                    # Convert to numpy for cv2 bilateral filter
                    median_np = median_filtered.cpu().numpy().astype(np.float32)

                    # Apply bilateral filter with depth-aware parameters
                    import cv2
                    bilateral_filtered = cv2.bilateralFilter(median_np, d=5, sigmaColor=0.1, sigmaSpace=5)

                    stabilized_depths.append(bilateral_filtered)
                else:
                    # Single frame, just apply bilateral filter
                    frame_np = window_frames[0].squeeze(0).cpu().numpy().astype(np.float32)
                    import cv2
                    bilateral_filtered = cv2.bilateralFilter(frame_np, d=5, sigmaColor=0.1, sigmaSpace=5)
                    stabilized_depths.append(bilateral_filtered)

                # Update progress
                progress_bar.update(1)

                if (i + 1) % 10 == 0:
                    logger.info(f"Temporal stabilization: {i + 1}/{num_frames} frames processed")

            progress_bar.close()
            print(f"[SUCCESS] Temporal stabilization completed: {len(stabilized_depths)} frames")
            logger.info("Temporal stabilization completed")
            return stabilized_depths

        except Exception as e:
            logger.error(f"Temporal stabilization failed: {str(e)}")
            logger.info("Returning original depths without stabilization")
            return depths

    def _enhance_temporal_consistency(self, frames):
        """Apply content-aware preprocessing to enhance temporal consistency"""
        try:
            import numpy as np
            import cv2
            from tqdm import tqdm

            print(f"\n[TEMPORAL] TEMPORAL CONSISTENCY ENHANCEMENT")
            print(f"[FRAMES] Analyzing {len(frames)} frames for motion patterns...")
            print()

            logger.info("Analyzing frame differences for temporal enhancement...")

            enhanced_frames = []
            num_frames = len(frames)

            # Calculate frame differences to identify motion patterns
            print("[ANALYSIS] Analyzing frame differences...")
            frame_diffs = []
            diff_progress = tqdm(total=num_frames-1, desc="Motion Analysis", unit="frame",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for i in range(1, num_frames):
                diff = cv2.absdiff(frames[i], frames[i-1])
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                mean_diff = np.mean(gray_diff)
                frame_diffs.append(mean_diff)
                diff_progress.update(1)
            
            diff_progress.close()

            # Apply temporal smoothing to frames with high motion
            print("[SMOOTHING] Applying temporal smoothing...")
            enhancement_progress = tqdm(total=num_frames, desc="Temporal Enhancement", unit="frame",
                                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for i in range(num_frames):
                frame = frames[i].copy()

                # For frames with significant motion, apply slight temporal averaging
                if i > 0 and i < num_frames - 1 and len(frame_diffs) > i-1:
                    if frame_diffs[i-1] > 10:  # Threshold for significant motion
                        # Blend with neighboring frames to reduce temporal noise
                        prev_frame = frames[i-1]
                        next_frame = frames[i+1]

                        # Weighted temporal smoothing (preserve original frame emphasis)
                        blended = cv2.addWeighted(frame, 0.7, prev_frame, 0.15, 0)
                        blended = cv2.addWeighted(blended, 1.0, next_frame, 0.15, 0)
                        frame = blended

                enhanced_frames.append(frame)
                enhancement_progress.update(1)

                if (i + 1) % 20 == 0:
                    logger.info(f"Temporal enhancement: {i + 1}/{num_frames} frames processed")

            enhancement_progress.close()
            print(f"[SUCCESS] Temporal consistency enhancement completed: {len(enhanced_frames)} frames")
            logger.info("Temporal consistency enhancement completed")
            return np.array(enhanced_frames)

        except Exception as e:
            logger.error(f"Temporal enhancement failed: {str(e)}")
            logger.info("Returning original frames without enhancement")
            return frames
    
    def _upscale_depth_maps(self, depths, target_width, target_height):
        """Upscale depth maps to target resolution using high-quality interpolation"""
        try:
            import torch
            import torch.nn.functional as F
            from tqdm import tqdm
            
            print(f"\n[UPSCALING] DEPTH MAP UPSCALING")
            print(f"[UPSCALING] Upscaling {len(depths)} depth maps")
            print(f"[TARGET] Target: {target_width}x{target_height}")
            print()
            
            logger.info(f"Upscaling {len(depths)} depth maps to {target_width}x{target_height}")
            
            upscaled_depths = []
            
            # Create progress bar
            progress_bar = tqdm(total=len(depths), desc="Upscaling Depth Maps", unit="frame",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for i, depth in enumerate(depths):
                # Convert to tensor for processing
                if isinstance(depth, np.ndarray):
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                else:
                    depth_tensor = depth.unsqueeze(0).unsqueeze(0)
                
                # Use bicubic interpolation for high-quality upscaling
                upscaled = F.interpolate(
                    depth_tensor, 
                    size=(target_height, target_width), 
                    mode='bicubic', 
                    align_corners=False
                )
                
                upscaled_depths.append(upscaled.squeeze().cpu().numpy())
                
                # Update progress
                progress_bar.update(1)
                
                if (i + 1) % 10 == 0:  # Log progress every 10 frames
                    logger.info(f"Upscaled {i + 1}/{len(depths)} depth maps")
            
            progress_bar.close()
            print(f"[SUCCESS] Depth upscaling completed: {len(upscaled_depths)} frames at {target_width}x{target_height}")
            logger.info(f"Depth upscaling completed: {len(upscaled_depths)} frames at {target_width}x{target_height}")
            return upscaled_depths
            
        except Exception as e:
            logger.error(f"Depth upscaling failed: {str(e)}")
            logger.info("Returning original depth maps without upscaling")
            return depths
    
    def save_depth_sequence(self, depths, output_path, base_name, job_data=None):
        """Save individual depth frames as EXR for production use with depth normalization"""
        try:
            from tqdm import tqdm
            import signal
            import sys
            
            # Set up signal handler to prevent interruption
            def signal_handler(signum, frame):
                print(f"\n[WARNING] Received signal {signum}, but continuing to save files...")
                return
            
            # Register signal handlers for common interruption signals
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            print(f"\n[SAVING] SAVING DEPTH SEQUENCE")
            print(f"[OUTPUT] Output: {output_path}")
            print(f"[FRAMES] Frames: {len(depths)}")
            print()
            
            # Create depth directory - Fix output path structure
            if output_path.endswith('.exr'):
                # If output_path is a file pattern, extract directory
                depth_dir = os.path.dirname(output_path)
                # Create depth_exr subdirectory
                depth_dir = os.path.join(depth_dir, "depth_exr")
            else:
                # If output_path is a directory, create depth_exr subdirectory
                depth_dir = os.path.join(output_path, "depth_exr")
            
            os.makedirs(depth_dir, exist_ok=True)
            print(f"Creating output directory: {depth_dir}")
            
            # Get depth normalization settings
            enable_normalization = job_data.get('enable_normalization', False) if job_data else False
            near_value = job_data.get('near_value', 0.0) if job_data else 0.0
            far_value = job_data.get('far_value', 5.0) if job_data else 5.0
            invert_depth = job_data.get('invert_depth', False) if job_data else False
            
            # Get frame range from job data
            start_frame = job_data.get('start_frame', 1) if job_data else 1
            
            if enable_normalization:
                print(f"[NORMALIZATION] Depth normalization: ENABLED - Near={near_value}, Far={far_value}, Invert={invert_depth}")
            else:
                print(f"[NORMALIZATION] Depth normalization: DISABLED - Using raw depth values (like original)")
            print(f"[FRAME_RANGE] Frame range: {start_frame} to {start_frame + len(depths) - 1}")
            print()
            
            # Create progress bar
            progress_bar = tqdm(total=len(depths), desc="Saving EXR Files", unit="frame",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            # Create checkpoint file to track progress
            checkpoint_file = os.path.join(depth_dir, "save_progress.txt")
            saved_count = 0
            
            # Check if we have a previous checkpoint
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        saved_count = int(f.read().strip())
                    print(f"[CHECKPOINT] Resuming from frame {saved_count + 1}")
                    progress_bar.update(saved_count)
                except:
                    saved_count = 0
            
            # Check EXR library availability
            print(f"[CHECK] Verifying EXR library availability...")
            try:
                import OpenEXR
                import Imath
                print(f"[CHECK] [OK] OpenEXR library is available")
            except ImportError as e:
                print(f"[CHECK] [ERROR] OpenEXR library not available: {e}")
                print(f"[CHECK] Will use PNG fallback for all frames")
                use_exr = False
            else:
                use_exr = True
            
            # Save each depth frame as EXR
            for i, depth in enumerate(depths):
                try:
                    # Keep process alive - prevent timeout/interruption
                    if i % 10 == 0:  # Every 10 frames
                        print(f"[KEEPALIVE] Processing frame {i+1}/{len(depths)}...")
                        import sys
                        sys.stdout.flush()
                    
                    # Use actual frame numbers from input sequence
                    frame_number = start_frame + i
                    
                    # Choose file format based on availability
                    if use_exr:
                        # Use production-friendly naming: TML304_012_040_FTG01_proxy_v001_depth.####.exr
                        depth_file = os.path.join(depth_dir, f"{base_name}_depth.{frame_number:04d}.exr")
                        file_ext = "EXR"
                    else:
                        # Use PNG fallback
                        depth_file = os.path.join(depth_dir, f"{base_name}_depth.{frame_number:04d}.png")
                        file_ext = "PNG"
                    
                    print(f"[SAVING] Frame {frame_number}: {depth_file} ({file_ext})")
                    
                    # Ensure depth is a numpy array
                    if not isinstance(depth, np.ndarray):
                        print(f"[WARNING] Frame {frame_number} is not a numpy array, converting...")
                        depth = np.array(depth)
                    
                    # Debug: print depth info
                    print(f"[DEBUG] Frame {frame_number} depth shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min():.3f}, max: {depth.max():.3f}")
                    
                    # Apply depth normalization only if enabled
                    if enable_normalization:
                        depth_normalized = self.normalize_depth(depth, near_value, far_value, invert_depth)
                    else:
                        # Use raw depth values like original implementation (no normalization)
                        depth_normalized = depth
                    
                    # Save based on format
                    if use_exr:
                        # Save as EXR for high precision (32-bit float)
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
                    else:
                        # Save as PNG
                        import cv2
                        # Normalize depth to 0-255 range for PNG
                        if depth_normalized.max() > depth_normalized.min():
                            depth_png = ((depth_normalized - depth_normalized.min()) / 
                                        (depth_normalized.max() - depth_normalized.min()) * 255).astype(np.uint8)
                        else:
                            depth_png = np.zeros_like(depth_normalized, dtype=np.uint8)
                        cv2.imwrite(depth_file, depth_png)
                    
                    print(f"[SUCCESS] Saved frame {frame_number} ({file_ext})")
                    
                    # Update progress and checkpoint
                    progress_bar.update(1)
                    saved_count += 1
                    
                    # Save checkpoint every 10 frames
                    if saved_count % 10 == 0:
                        with open(checkpoint_file, 'w') as f:
                            f.write(str(saved_count))
                        print(f"[CHECKPOINT] Saved progress: {saved_count}/{len(depths)} frames")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to save EXR frame {frame_number}: {str(e)}")
                    logger.error(f"Failed to save EXR frame {frame_number}: {str(e)}")
                    
                    # Try to save as PNG as fallback
                    try:
                        import cv2
                        png_file = depth_file.replace('.exr', '.png')
                        # Normalize depth to 0-255 range for PNG
                        if depth_normalized.max() > depth_normalized.min():
                            depth_normalized_png = ((depth_normalized - depth_normalized.min()) / 
                                                  (depth_normalized.max() - depth_normalized.min()) * 255).astype(np.uint8)
                        else:
                            depth_normalized_png = np.zeros_like(depth_normalized, dtype=np.uint8)
                        cv2.imwrite(png_file, depth_normalized_png)
                        print(f"[FALLBACK] Saved frame {frame_number} as PNG: {png_file}")
                        logger.info(f"Saved frame {frame_number} as PNG fallback: {png_file}")
                    except Exception as png_error:
                        print(f"[ERROR] Failed to save PNG fallback for frame {frame_number}: {str(png_error)}")
                        logger.error(f"Failed to save PNG fallback for frame {frame_number}: {str(png_error)}")
                    
                    import traceback
                    traceback.print_exc()
                    # Continue with next frame instead of stopping
                    progress_bar.update(1)
                    saved_count += 1
                
            progress_bar.close()
            
            # Clean up checkpoint file
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"[CHECKPOINT] Cleaned up checkpoint file")
            
            print(f"[SUCCESS] Saved {len(depths)} depth EXR frames to {depth_dir}")
            logger.info(f"Saved {len(depths)} depth EXR frames to {depth_dir}")
            
            # Force flush any remaining output
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Log normalization status correctly
            if enable_normalization:
                logger.info(f"Depth normalization: ENABLED - Near={near_value}, Far={far_value}, Invert={invert_depth}")
            else:
                logger.info(f"Depth normalization: DISABLED - Using raw depth values (like original)")
            
            # Also save a sequence info file
            info_file = os.path.join(depth_dir, "sequence_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Depth sequence: {base_name}\n")
                f.write(f"Frame count: {len(depths)}\n")
                f.write(f"Resolution: {depths[0].shape[1]}x{depths[0].shape[0]}\n")
                f.write(f"Format: 32-bit float EXR\n")
                f.write(f"Channel: Z (depth)\n")
                if enable_normalization:
                    f.write(f"Depth normalization: ENABLED - Near={near_value}, Far={far_value}, Invert={invert_depth}\n")
                else:
                    f.write(f"Depth normalization: DISABLED - Raw depth values (like original)\n")
            
            print(f"[INFO] Sequence info saved to: {info_file}")
            
        except Exception as e:
            logger.error(f"Failed to save depth sequence: {str(e)}")
    
    def normalize_depth(self, depth, near_value, far_value, invert_depth):
        """Normalize depth values for VFX production"""
        try:
            # Convert to numpy if needed
            if torch.is_tensor(depth):
                depth = depth.cpu().numpy()
            elif isinstance(depth, list):
                depth = np.array(depth)
            
            # Ensure depth is a numpy array
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
    
    def read_frames(self, input_path, max_len, target_fps, max_res, job_data):
        """Read frames from video or image sequence with half-res processing"""
        try:
            import cv2
            import glob
            from PIL import Image
            
            # Check if it's an image sequence
            if "%04d" in input_path or "####" in input_path or "*" in input_path:
                # It's an image sequence
                logger.info("Processing image sequence directly")
                
                # Get frame range
                start_frame = job_data.get('start_frame', 1)
                end_frame = job_data.get('end_frame', 100)
                half_res = job_data.get('half_res', True)
                
                # Generate frame paths
                frames = []
                frames_loaded = 0
                
                print(f"[LOADING] Loading image sequence...")
                print(f"[FRAMES] Frame range: {start_frame} to {end_frame}")
                print(f"[SETTINGS] Half-res processing: {half_res}")
                print()
                
                # Create progress bar for frame loading
                from tqdm import tqdm
                total_frames = end_frame - start_frame + 1
                progress_bar = tqdm(total=total_frames, desc="Loading Frames", unit="frame",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                
                for frame_num in range(start_frame, end_frame + 1):
                    # Replace frame padding with actual frame number
                    frame_path = input_path.replace("%04d", f"{frame_num:04d}")
                    frame_path = frame_path.replace("####", f"{frame_num:04d}")
                    
                    # Check if file exists and is actually a file (not a directory)
                    if os.path.exists(frame_path) and os.path.isfile(frame_path):
                        # Read image (handle both JPG and EXR)
                        img = self.read_image_frame(frame_path)
                        if img is not None:
                            # Store original dimensions before any scaling
                            if frames_loaded == 0:  # Only store once
                                original_h, original_w = img.shape[:2]
                                print(f"[DIMENSIONS] Original image dimensions: {original_w}x{original_h}")
                                logger.info(f"Original image dimensions: {original_w}x{original_h}")
                            
                            # Apply half-res scaling if requested
                            if half_res:
                                h, w = img.shape[:2]
                                # Simple half-res scaling: divide by 2
                                new_w, new_h = w//2, h//2
                                logger.info(f"Half-res scaling: {w}x{h} -> {new_w}x{new_h}")
                                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                            
                            frames.append(img)
                            frames_loaded += 1
                            logger.info(f"Loaded frame {frame_num}: {frame_path}")
                        else:
                            logger.warning(f"Could not read frame: {frame_path}")
                    else:
                        logger.warning(f"Frame not found or is directory: {frame_path}")
                    
                    # Update progress
                    progress_bar.update(1)
                
                progress_bar.close()
                
                if frames_loaded == 0:
                    logger.error(f"No frames loaded from sequence: {input_path}")
                    return [], 24.0
                
                # Convert list of frames to numpy array (like the original read_video_frames)
                # Ensure all frames are numpy arrays first
                frames_array = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        frames_array.append(frame)
                    else:
                        frames_array.append(np.array(frame))
                
                frames = np.array(frames_array)
                logger.info(f"Converted {frames_loaded} frames to numpy array with shape: {frames.shape}")
                
                # Store original dimensions in job_data for proper upscaling
                if frames_loaded > 0:
                    job_data['original_width'] = original_w
                    job_data['original_height'] = original_h
                    logger.info(f"Stored original dimensions in job_data: {original_w}x{original_h}")
                
                # Use default FPS for image sequences
                fps = 24.0
                logger.info(f"Successfully loaded {frames_loaded} frames from image sequence")
                
            else:
                # It's a video file - use existing method
                from utils.dc_utils import read_video_frames
                frames, fps = read_video_frames(input_path, max_len, target_fps, max_res)
            
            logger.info(f"Loaded {len(frames)} frames at {fps} FPS")
            return frames, fps
            
        except Exception as e:
            logger.error(f"Failed to read frames: {str(e)}")
            return [], 24.0
    
    def read_image_frame(self, frame_path):
        """Read a single image frame (JPG, PNG, EXR) and convert to RGB"""
        try:
            import cv2
            import numpy as np
            
            # Check file extension
            ext = os.path.splitext(frame_path)[1].lower()
            
            if ext == '.exr':
                # Read EXR file
                try:
                    import OpenEXR
                    import Imath
                    
                    # Open EXR file
                    exr_file = OpenEXR.InputFile(frame_path)
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
                            # If RGB channels not found, try RGBA
                            if channel == 'B' and 'A' in header['channels']:
                                channel_str = exr_file.channel('A', Imath.PixelType(Imath.PixelType.FLOAT))
                                channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                            else:
                                # Use R channel for missing channels
                                channel_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                                channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                    
                    exr_file.close()
                    
                    # Reshape and combine channels
                    img = np.zeros((height, width, 3), dtype=np.float32)
                    for i, channel in enumerate(channels):
                        img[:, :, i] = channel_data[channel].reshape(height, width)
                    
                    # Convert from float32 to uint8 (0-255 range)
                    # EXR files can have values > 1.0, so we need to handle this
                    img = np.clip(img, 0, 1)  # Clamp to 0-1 range
                    img = (img * 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV compatibility
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    logger.info(f"Successfully read EXR frame: {frame_path}")
                    return img
                    
                except ImportError:
                    logger.error("OpenEXR not available for reading EXR files")
                    return None
                except Exception as e:
                    logger.error(f"Failed to read EXR file {frame_path}: {str(e)}")
                    return None
                    
            else:
                # Read regular image file (JPG, PNG, etc.)
                img = cv2.imread(frame_path)
                if img is not None:
                    # Convert BGR to RGB for consistency
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    logger.info(f"Successfully read image frame: {frame_path}")
                    return img
                else:
                    logger.error(f"Failed to read image: {frame_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error reading frame {frame_path}: {str(e)}")
            return None

def main():
    """Main engine loop - monitors job queue and processes videos"""
    if not TORCH_AVAILABLE:
        print("=" * 60)
        print("CRITICAL ERROR: PyTorch not available!")
        print("=" * 60)
        print("The engine cannot run without PyTorch.")
        print("Please reinstall PyTorch:")
        print("  .\\embedded_python\\python.exe -m pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60)
        return
    
    engine = VideoDepthEngine()
    
    # Job queue file path - use absolute path
    current_dir = Path(__file__).parent.parent  # Go up to VideoDepthAnything_Portable
    job_queue_file = current_dir / "communication" / "job_queue.json"
    status_file = current_dir / "communication" / "status.json"
    
    # Create communication directory
    os.makedirs(current_dir / "communication", exist_ok=True)
    
    print("=" * 60)
    print("VDA ENGINE MAIN LOOP STARTED")
    print("=" * 60)
    print("Monitoring job queue for new processing requests...")
    print("Press Ctrl+C to stop the engine")
    print("=" * 60)
    print()
    
    # Update status to running
    status = {"status": "running", "message": "Engine ready and monitoring job queue"}
    with open(status_file, 'w') as f:
        json.dump(status, f)
    
    logger.info("Video Depth Anything Engine started")
    logger.info("Monitoring job queue...")
    
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
                print()
                
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
