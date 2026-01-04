"""Workload managers for CPU, GPU, and Coral inference."""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .docker_manager import DockerWorkloadManager

logger = logging.getLogger(__name__)


class CPUWorkloadManager(DockerWorkloadManager):
    """Docker manager for CPU-only inference workloads."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize CPU workload manager."""
        super().__init__(model_path=model_path, **kwargs)

    def get_container_config(self) -> Dict[str, Any]:
        """Get container configuration for CPU workloads."""
        return {
            "cpu_count": os.cpu_count(),
            "mem_limit": "4g",
            "network_mode": "host",
        }


class GPUWorkloadManager(DockerWorkloadManager):
    """Docker manager for GPU (CUDA) inference workloads."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize GPU workload manager."""
        super().__init__(model_path=model_path, **kwargs)

    def get_container_config(self) -> Dict[str, Any]:
        """Get container configuration for GPU workloads with CUDA support."""
        import docker.types

        return {
            "runtime": "nvidia",  # Use nvidia runtime for GPU access
            "device_requests": [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            "environment": {
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "USE_TENSORFLOW_GPU": "true",  # Enable TensorFlow GPU acceleration
            },
            "mem_limit": "8g",
            "network_mode": "host",
        }


class CoralWorkloadManager(DockerWorkloadManager):
    """Docker manager for Google Coral inference workloads (standard driver)."""

    def __init__(self, coral_device: Optional[str] = None, model_path: Optional[str] = None, **kwargs):
        """
        Initialize Coral workload manager.

        Args:
            coral_device: Path to Coral device (e.g., '/dev/apex_0' or '/dev/bus/usb')
            model_path: Path to model file (will be mounted in container)
        """
        super().__init__(model_path=model_path, **kwargs)
        self.coral_device = coral_device or self._detect_coral_device()

    def _detect_coral_device(self) -> Optional[str]:
        """Try to detect Coral device automatically."""
        # Common Coral device paths
        possible_paths = [
            "/dev/apex_0",
            "/dev/bus/usb",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"Detected Coral device at: {path}")
                return path
        
        logger.warning("Could not auto-detect Coral device. You may need to specify it manually.")
        return "/dev/bus/usb"  # Default fallback

    def get_container_config(self) -> Dict[str, Any]:
        """Get container configuration for Coral workloads."""
        devices = []
        volumes = {}

        if self.coral_device:
            if Path(self.coral_device).is_dir():
                # If it's a directory (like /dev/bus/usb), mount it as a volume
                volumes[self.coral_device] = {"bind": self.coral_device, "mode": "rw"}
            elif Path(self.coral_device).is_file():
                # If it's a device file, add it as a device
                devices.append(self.coral_device)

        return {
            "devices": devices,
            "volumes": volumes,
            "privileged": True,  # Coral often needs privileged access
            "mem_limit": "2g",
            "network_mode": "host",
            "environment": {
                "PYTHONUNBUFFERED": "1",
            },
        }


class CoralMaxWorkloadManager(CoralWorkloadManager):
    """Docker manager for Google Coral inference workloads (max performance driver)."""

    def __init__(self, coral_device: Optional[str] = None, model_path: Optional[str] = None, **kwargs):
        """
        Initialize Coral Max workload manager.
        
        Uses libedgetpu1-max (500 MHz) instead of libedgetpu1-std (250 MHz).
        Can run hot but provides maximum performance.

        Args:
            coral_device: Path to Coral device (e.g., '/dev/apex_0' or '/dev/bus/usb')
            model_path: Path to model file (will be mounted in container)
        """
        # Set image name to use Dockerfile.max
        # The image will be built from Dockerfile.max
        if "image_name" not in kwargs:
            kwargs["image_name"] = "llms-fpga-benchmark-max:latest"
        super().__init__(coral_device=coral_device, model_path=model_path, **kwargs)

