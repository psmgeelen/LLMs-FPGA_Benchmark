"""Example usage of the LLMs-FPGA benchmark tool."""

import logging
from pathlib import Path

from src.llms_fpga_benchmark.model_loader import ModelLoader
from src.llms_fpga_benchmark.workloads import CPUWorkloadManager, GPUWorkloadManager, CoralWorkloadManager
from src.llms_fpga_benchmark.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_cpu_benchmark():
    """Example: Run CPU benchmark."""
    model_path = "models/your_model.tflite"  # Replace with your model path
    
    # Load model
    model_loader = ModelLoader(model_path)
    
    # Create CPU workload manager
    workload_manager = CPUWorkloadManager(
        image_name="llms-fpga-benchmark:latest",
        model_path=model_path,
    )
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(
        workload_manager=workload_manager,
        model_loader=model_loader,
        num_iterations=10,
        warmup_iterations=2,
    )
    
    # Run benchmark
    with workload_manager:
        results = benchmark_runner.run_benchmark()
    
    print("CPU Benchmark Results:")
    print(f"  Mean inference time: {results['inference_times']['mean']:.4f}s")
    print(f"  Mean CPU usage: {results['cpu_usage']['mean']:.2f}%")
    print(f"  Mean milicpu-seconds: {results['milicpu_seconds']['mean']:.2f}")
    
    return results


def example_gpu_benchmark():
    """Example: Run GPU benchmark."""
    model_path = "models/your_model.tflite"  # Replace with your model path
    
    # Load model
    model_loader = ModelLoader(model_path)
    
    # Create GPU workload manager
    workload_manager = GPUWorkloadManager(
        image_name="llms-fpga-benchmark:latest",
        model_path=model_path,
    )
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(
        workload_manager=workload_manager,
        model_loader=model_loader,
        num_iterations=10,
        warmup_iterations=2,
    )
    
    # Run benchmark
    with workload_manager:
        results = benchmark_runner.run_benchmark()
    
    print("GPU Benchmark Results:")
    print(f"  Mean inference time: {results['inference_times']['mean']:.4f}s")
    print(f"  Mean CPU usage: {results['cpu_usage']['mean']:.2f}%")
    if 'miligpu_seconds' in results:
        print(f"  Mean miligpu-seconds: {results['miligpu_seconds']['mean']:.2f}")
    if 'gpu_usage' in results:
        print(f"  Mean GPU utilization: {results['gpu_usage']['mean']:.2f}%")
    
    return results


def example_coral_benchmark():
    """Example: Run Coral benchmark."""
    model_path = "models/your_model.tflite"  # Replace with your model path
    
    # Load model
    model_loader = ModelLoader(model_path)
    
    # Create Coral workload manager
    workload_manager = CoralWorkloadManager(
        image_name="llms-fpga-benchmark:latest",
        model_path=model_path,
        coral_device="/dev/apex_0",  # Adjust based on your setup
    )
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(
        workload_manager=workload_manager,
        model_loader=model_loader,
        num_iterations=10,
        warmup_iterations=2,
    )
    
    # Run benchmark
    with workload_manager:
        results = benchmark_runner.run_benchmark()
    
    print("Coral Benchmark Results:")
    print(f"  Mean inference time: {results['inference_times']['mean']:.4f}s")
    print(f"  Mean CPU usage: {results['cpu_usage']['mean']:.2f}%")
    print(f"  Mean milicpu-seconds: {results['milicpu_seconds']['mean']:.2f}")
    
    return results


if __name__ == "__main__":
    # Uncomment the benchmark you want to run:
    # example_cpu_benchmark()
    # example_gpu_benchmark()
    # example_coral_benchmark()
    pass

