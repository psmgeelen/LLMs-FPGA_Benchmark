"""Base Docker management class for running inference workloads."""

import docker
import logging
import time
from typing import Dict, Optional, List, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DockerWorkloadManager(ABC):
    """Base class for managing Docker containers for inference workloads."""

    def __init__(
        self,
        image_name: str = "llms-fpga-benchmark:latest",
        container_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the Docker workload manager.

        Args:
            image_name: Name of the Docker image to use
            container_name: Optional name for the container (auto-generated if None)
            model_path: Path to model file (will be mounted in container)
        """
        self.client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name or f"{self.__class__.__name__.lower()}-{int(time.time())}"
        self.model_path = model_path
        self.container: Optional[docker.models.containers.Container] = None
        logger.info(f"Initialized {self.__class__.__name__} with image: {image_name}")

    def ensure_image_exists(self) -> None:
        """Ensure the Docker image exists, build if necessary."""
        try:
            self.client.images.get(self.image_name)
            logger.info(f"Image {self.image_name} already exists")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image {self.image_name} not found. Please build it first using: docker build -t {self.image_name} .")

    @abstractmethod
    def get_container_config(self) -> Dict[str, Any]:
        """
        Get the container configuration for this workload type.
        
        Returns:
            Dictionary with container configuration (device_requests, volumes, etc.)
        """
        pass

    def start_container(self) -> None:
        """Start the Docker container with the appropriate configuration."""
        if self.container is not None:
            logger.warning("Container already running")
            return

        self.ensure_image_exists()
        config = self.get_container_config()
        
        # Add model volume mount if model_path is provided
        if self.model_path:
            from pathlib import Path
            model_path = Path(self.model_path)
            if model_path.exists():
                # Mount the model file or directory to /app/models in container
                model_dir = model_path.parent
                model_file = model_path.name
                container_model_path = f"/app/models/{model_file}"
                
                # Ensure volumes dict exists
                if "volumes" not in config:
                    config["volumes"] = {}
                
                # Mount the model directory
                config["volumes"][str(model_dir.absolute())] = {
                    "bind": "/app/models",
                    "mode": "ro"
                }
                logger.info(f"Mounted model: {model_path} -> {container_model_path}")
        
        try:
            # Start container with a keep-alive command to prevent it from exiting
            # We'll use exec_run for actual commands
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                command=["tail", "-f", "/dev/null"],  # Keep container running
                detach=True,
                **config
            )
            logger.info(f"Started container: {self.container_name} (ID: {self.container.short_id})")
        except docker.errors.ContainerError as e:
            logger.error(f"Failed to start container: {e}")
            raise
        except docker.errors.ImageNotFound:
            logger.error(f"Image {self.image_name} not found. Please build it first.")
            raise

    def stop_container(self) -> None:
        """Stop and remove the Docker container."""
        if self.container is None:
            return

        try:
            self.container.stop()
            self.container.remove()
            logger.info(f"Stopped and removed container: {self.container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container {self.container_name} not found")
        except Exception as e:
            logger.error(f"Error stopping container: {e}")
        finally:
            self.container = None

    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a command in the running container.

        Args:
            command: Command to execute
            **kwargs: Additional arguments for exec_run

        Returns:
            Dictionary with exit_code, output, and execution_time
        """
        if self.container is None:
            raise RuntimeError("Container is not running. Call start_container() first.")

        start_time = time.time()
        result = self.container.exec_run(command, **kwargs)
        execution_time = time.time() - start_time

        return {
            "exit_code": result.exit_code,
            "output": result.output.decode("utf-8") if result.output else "",
            "execution_time": execution_time,
        }

    def get_container_stats(self) -> Dict[str, Any]:
        """
        Get current container resource usage statistics.

        Returns:
            Dictionary with CPU, memory, and other resource stats
        """
        if self.container is None:
            raise RuntimeError("Container is not running. Call start_container() first.")

        stats = self.container.stats(stream=False)
        return {
            "cpu_usage": stats.get("cpu_stats", {}),
            "memory_usage": stats.get("memory_stats", {}),
            "network": stats.get("networks", {}),
            "timestamp": stats.get("read", ""),
        }

    def __enter__(self):
        """Context manager entry."""
        self.start_container()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_container()

