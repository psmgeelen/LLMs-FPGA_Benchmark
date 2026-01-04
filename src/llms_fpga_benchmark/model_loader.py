"""Model loader for TensorFlow Lite transformer models and Hugging Face models."""

import os
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages TensorFlow Lite models for inference."""

    def __init__(self, model_path: str):
        """
        Initialize the model loader.

        Args:
            model_path: Path to the TensorFlow Lite model file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_size = self.model_path.stat().st_size
        logger.info(f"Loaded model: {self.model_path} ({self.model_size / 1024 / 1024:.2f} MB)")

    def get_model_path(self) -> str:
        """Get the absolute path to the model file."""
        return str(self.model_path.absolute())

    def get_model_size(self) -> int:
        """Get the model file size in bytes."""
        return self.model_size

    def validate_model(self) -> bool:
        """
        Validate that the model file is a valid TensorFlow Lite model.
        
        Returns:
            True if the model appears valid
        """
        # Basic validation: check file extension and that it's not empty
        if self.model_path.suffix not in ['.tflite', '.lite']:
            logger.warning(f"Model file doesn't have .tflite extension: {self.model_path}")
        
        if self.model_size == 0:
            logger.error("Model file is empty")
            return False
        
        return True

