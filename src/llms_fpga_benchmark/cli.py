"""Command-line interface for running benchmarks."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .model_loader import ModelLoader
from .workloads import CPUWorkloadManager, GPUWorkloadManager, CoralWorkloadManager, CoralMaxWorkloadManager
from .benchmark import BenchmarkRunner
from .model_converter import ModelConverter
from .prompt_generator import PromptGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model inference on CPU, GPU, or Coral"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to TensorFlow Lite model file"
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["cpu", "gpu", "coral", "coral-max"],
        required=True,
        help="Workload type to benchmark (coral-max uses max performance driver)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results (default: stdout)"
    )
    parser.add_argument(
        "--coral-device",
        type=str,
        help="Path to Coral device (e.g., /dev/apex_0)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="llms-fpga-benchmark:latest",
        help="Docker image name (default: llms-fpga-benchmark:latest)"
    )
    parser.add_argument(
        "--convert-tiny-llm",
        action="store_true",
        help="Automatically download and convert Tiny-LLM from Hugging Face"
    )
    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help="Use generated prompts for text generation benchmarks"
    )
    parser.add_argument(
        "--model-params",
        type=int,
        default=10_000_000,
        help="Number of model parameters (for TOPS calculation, default: 10M for Tiny-LLM)"
    )

    args = parser.parse_args()

    # Convert Tiny-LLM if requested
    model_path = args.model
    if args.convert_tiny_llm:
        logger.info("Converting Tiny-LLM from Hugging Face...")
        converter = ModelConverter()
        output_path = Path(args.model)
        if not output_path.exists():
            try:
                model_path = converter.download_and_convert_tiny_llm(
                    output_path=str(output_path),
                    quantize=True,  # Quantize for Coral compatibility
                )
                logger.info(f"Model converted and saved to: {model_path}")
            except Exception as e:
                logger.error(f"Failed to convert Tiny-LLM: {e}")
                sys.exit(1)
        else:
            logger.info(f"Model already exists at {model_path}, skipping conversion")

    # Load model
    try:
        model_loader = ModelLoader(model_path)
        model_loader.validate_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create workload manager
    workload_kwargs = {
        "model_path": model_path,
    }
    
    # Set image name based on workload type
    if args.workload == "coral-max":
        # Use max image for coral-max workload
        workload_kwargs["image_name"] = args.image.replace(":latest", "-max:latest") if args.image == "llms-fpga-benchmark:latest" else args.image
    else:
        workload_kwargs["image_name"] = args.image
    
    if args.workload in ["coral", "coral-max"] and args.coral_device:
        workload_kwargs["coral_device"] = args.coral_device

    workload_managers = {
        "cpu": CPUWorkloadManager,
        "gpu": GPUWorkloadManager,
        "coral": CoralWorkloadManager,
        "coral-max": CoralMaxWorkloadManager,
    }

    workload_manager = workload_managers[args.workload](**workload_kwargs)

    # Create prompt generator if requested
    prompt_generator = None
    if args.use_prompts:
        prompt_generator = PromptGenerator()
        logger.info(f"Using {prompt_generator.get_prompt_count()} prompts for benchmarking")

    # Run benchmark
    try:
        benchmark_runner = BenchmarkRunner(
            workload_manager=workload_manager,
            model_loader=model_loader,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            prompt_generator=prompt_generator,
            model_params=args.model_params,
        )

        with workload_manager:
            results = benchmark_runner.run_benchmark()

        # Output results
        results_json = json.dumps(results, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(results_json)
            logger.info(f"Results saved to {args.output}")
        else:
            print(results_json)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

