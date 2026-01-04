"""Prompt generator for benchmarking LLM inference."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates test prompts for LLM benchmarking."""

    DEFAULT_PROMPTS = [
        "According to all known laws of aviation, there is no way a bee should be able to fly.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "In a world where artificial intelligence becomes increasingly sophisticated,",
        "Once upon a time, in a galaxy far, far away, there lived a",
        "The future of technology lies in the intersection of",
        "Climate change is one of the most pressing issues facing",
        "The art of programming requires both logical thinking and",
        "Machine learning models have revolutionized the way we",
        "Quantum computing represents a paradigm shift in",
        "The relationship between humans and artificial intelligence is",
    ]

    def __init__(self, prompts: List[str] = None):
        """
        Initialize the prompt generator.

        Args:
            prompts: Custom list of prompts. If None, uses default prompts.
        """
        self.prompts = prompts or self.DEFAULT_PROMPTS
        if len(self.prompts) < 10:
            logger.warning(f"Only {len(self.prompts)} prompts provided, expected 10")
        logger.info(f"Initialized PromptGenerator with {len(self.prompts)} prompts")

    def get_prompts(self) -> List[str]:
        """Get all prompts."""
        return self.prompts

    def get_prompt(self, index: int) -> str:
        """
        Get a specific prompt by index.

        Args:
            index: Prompt index (0-based)

        Returns:
            The prompt at the given index
        """
        if index < 0 or index >= len(self.prompts):
            raise IndexError(f"Prompt index {index} out of range (0-{len(self.prompts)-1})")
        return self.prompts[index]

    def get_prompt_count(self) -> int:
        """Get the number of available prompts."""
        return len(self.prompts)

