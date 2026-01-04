"""Benchmarking logic for measuring speed and resource consumption."""

import time
import logging
import statistics
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

from .docker_manager import DockerWorkloadManager
from .model_loader import ModelLoader
from .workloads import GPUWorkloadManager, CoralWorkloadManager, CoralMaxWorkloadManager
from .prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks for inference workloads."""

    def __init__(
        self,
        workload_manager: DockerWorkloadManager,
        model_loader: ModelLoader,
        num_iterations: int = 100,
        warmup_iterations: int = 2,
        prompt_generator: Optional[PromptGenerator] = None,
        model_params: Optional[int] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            workload_manager: Docker workload manager instance
            model_loader: Model loader instance
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations before benchmarking
            prompt_generator: Optional prompt generator for text generation benchmarks
            model_params: Number of model parameters (for TOPS calculation)
        """
        self.workload_manager = workload_manager
        self.model_loader = model_loader
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.prompt_generator = prompt_generator
        self.model_params = model_params or 10_000_000  # Default to 10M for Tiny-LLM
        self.is_gpu_workload = isinstance(self.workload_manager, GPUWorkloadManager)

    def _calculate_cpu_usage(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from container stats."""
        cpu_stats = stats.get("cpu_usage", {})
        if not cpu_stats:
            return 0.0

        # Calculate CPU percentage
        cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        system_delta = cpu_stats.get("system_cpu_usage", 0)
        
        if system_delta == 0:
            return 0.0

        # Get number of CPUs
        online_cpus = cpu_stats.get("online_cpus", 1)
        
        cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0
        return cpu_percent

    def _calculate_memory_usage(self, stats: Dict[str, Any]) -> int:
        """Calculate memory usage in bytes from container stats."""
        memory_stats = stats.get("memory_usage", {})
        return memory_stats.get("usage", 0)

    def _get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage statistics using nvidia-ml-py."""
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            return {
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "memory_used_mb": mem_info.used / 1024 / 1024,
                "memory_total_mb": mem_info.total / 1024 / 1024,
                "power_watts": power,
            }
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
            return None

    def _run_inference_command(self, iteration: int) -> Dict[str, Any]:
        """Run a single inference iteration in the container (for warmup)."""
        # Model is mounted at /app/models/, so we need just the filename
        from pathlib import Path
        model_filename = Path(self.model_loader.get_model_path()).name
        container_model_path = f"/app/models/{model_filename}"
        
        # Check if this is a Coral workload
        use_coral = isinstance(self.workload_manager, CoralWorkloadManager)
        coral_flag = " --coral" if use_coral else ""
        
        # Get prompt if available
        prompt_flag = ""
        if self.prompt_generator:
            prompt_index = iteration % self.prompt_generator.get_prompt_count()
            prompt = self.prompt_generator.get_prompt(prompt_index)
            # Escape quotes for shell command
            prompt_escaped = prompt.replace('"', '\\"')
            prompt_flag = f' --prompt "{prompt_escaped}"'
        
        # Command to run inference (this will be implemented in the container)
        command = f"python /app/run_inference.py --model {container_model_path} --iteration {iteration}{coral_flag}{prompt_flag}"
        
        result = self.workload_manager.execute_command(command)
        
        # Get resource stats during execution
        stats = self.workload_manager.get_container_stats()
        
        return {
            "iteration": iteration,
            "exit_code": result["exit_code"],
            "output": result["output"],
            "execution_time": result["execution_time"],
            "cpu_usage_percent": self._calculate_cpu_usage(stats),
            "memory_usage_bytes": self._calculate_memory_usage(stats),
            "gpu_stats": self._get_gpu_usage() if isinstance(self.workload_manager, GPUWorkloadManager) else None,
        }
    
    def _run_batch_inference(self, start_iteration: int, num_iterations: int) -> Dict[str, Any]:
        """
        Run multiple inference iterations in batch mode (model loaded once).
        
        Args:
            start_iteration: Starting iteration number
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with batch results including per-iteration times
        """
        from pathlib import Path
        
        model_filename = Path(self.model_loader.get_model_path()).name
        container_model_path = f"/app/models/{model_filename}"
        
        # Check if this is a Coral workload
        use_coral = isinstance(self.workload_manager, CoralWorkloadManager)
        coral_flag = " --coral" if use_coral else ""
        
        # Prepare prompts for batch mode
        prompts_list = []
        if self.prompt_generator:
            for i in range(num_iterations):
                prompt_index = (start_iteration + i) % self.prompt_generator.get_prompt_count()
                prompts_list.append(self.prompt_generator.get_prompt(prompt_index))
        
        # Build command with batch mode
        command = f"python /app/run_inference.py --model {container_model_path} --iteration {start_iteration} --iterations {num_iterations}{coral_flag}"
        
        if prompts_list:
            # Escape prompts and join with commas
            prompts_escaped = [p.replace('"', '\\"') for p in prompts_list]
            prompts_str = ','.join(prompts_escaped)
            command += f' --prompts "{prompts_str}"'
        
        result = self.workload_manager.execute_command(command)
        
        # Get resource stats during execution
        stats = self.workload_manager.get_container_stats()
        
        # Parse individual iteration times and GPU stats from output
        # Look for lines like: "[Iteration X] Inference completed in Y.XXXXs GPU_USED=True/False GPU_UTIL=Z%"
        iteration_times = []
        gpu_utilizations = []
        gpu_used = False
        if result["output"]:
            # Parse inference times
            time_pattern = r'\[Iteration (\d+)\].*?inference completed in ([\d.]+)s'
            matches = re.findall(time_pattern, result["output"], re.IGNORECASE)
            for iter_num, time_str in matches:
                try:
                    iteration_times.append(float(time_str))
                except ValueError:
                    pass
            
            # Parse GPU usage flag - check for final GPU_USED value (after inference detection)
            # Look for GPU_USED_DETECTED first (more reliable), then fall back to GPU_USED
            gpu_detected_match = re.search(r'GPU_USED_DETECTED=(True|False)', result["output"])
            if gpu_detected_match:
                gpu_used = gpu_detected_match.group(1) == "True"
            else:
                # Fall back to GPU_USED flag (check last occurrence, which is after inference)
                gpu_used_matches = re.findall(r'GPU_USED=(True|False)', result["output"])
                if gpu_used_matches:
                    # Use the last occurrence (after inference detection)
                    gpu_used = gpu_used_matches[-1] == "True"
            
            # Parse GPU utilization from logs
            gpu_util_pattern = r'GPU_UTIL=([\d.]+)%'
            gpu_util_matches = re.findall(gpu_util_pattern, result["output"])
            for util_str in gpu_util_matches:
                try:
                    gpu_utilizations.append(float(util_str))
                except ValueError:
                    pass
        
        # If we couldn't parse times, use total time divided by iterations
        if not iteration_times and result["execution_time"] > 0:
            avg_time = result["execution_time"] / num_iterations
            iteration_times = [avg_time] * num_iterations
        
        return {
            "exit_code": result["exit_code"],
            "output": result["output"],
            "total_execution_time": result["execution_time"],
            "iteration_times": iteration_times,
            "gpu_utilizations": gpu_utilizations,
            "gpu_used": gpu_used,
            "cpu_usage_percent": self._calculate_cpu_usage(stats),
            "memory_usage_bytes": self._calculate_memory_usage(stats),
            "gpu_stats": self._get_gpu_usage() if isinstance(self.workload_manager, GPUWorkloadManager) else None,
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.

        Returns:
            Dictionary with benchmark results including:
            - inference_times: List of inference times
            - cpu_usage: CPU usage statistics
            - memory_usage: Memory usage statistics
            - gpu_usage: GPU usage statistics (if applicable)
            - milicpu_seconds: CPU time in millicpu-seconds
            - miligpu_seconds: GPU time in milligpu-seconds (if applicable)
        """
        logger.info(f"Starting benchmark with {self.num_iterations} iterations")
        
        # Warmup (run individually to warm up the system)
        logger.info(f"Running {self.warmup_iterations} warmup iterations...")
        for i in range(self.warmup_iterations):
            self._run_inference_command(i)
            time.sleep(0.5)  # Brief pause between iterations

        # Actual benchmark - run in batch mode (model loaded once)
        logger.info(f"Running {self.num_iterations} benchmark iterations in batch mode (model loaded once)...")
        batch_result = self._run_batch_inference(0, self.num_iterations)
        
        # Convert batch result to per-iteration results format
        results = []
        iteration_times = batch_result.get("iteration_times", [])
        if not iteration_times:
            # Fallback: use total time divided by iterations
            avg_time = batch_result["total_execution_time"] / self.num_iterations
            iteration_times = [avg_time] * self.num_iterations
        
        # Get GPU utilization per iteration if available
        gpu_utils = batch_result.get("gpu_utilizations", [])
        
        for i, inf_time in enumerate(iteration_times):
            # Get GPU utilization for this iteration if available
            gpu_util = gpu_utils[i] if i < len(gpu_utils) else None
            
            results.append({
                "iteration": i,
                "exit_code": batch_result["exit_code"],
                "output": batch_result["output"] if i == 0 else "",  # Only include output once
                "execution_time": inf_time,
                "cpu_usage_percent": batch_result["cpu_usage_percent"],
                "memory_usage_bytes": batch_result["memory_usage_bytes"],
                "gpu_stats": batch_result["gpu_stats"],
                "gpu_used": batch_result.get("gpu_used", False),
                "gpu_utilization": gpu_util,
            })

        # Aggregate results
        inference_times = [r["execution_time"] for r in results]
        cpu_usages = [r["cpu_usage_percent"] for r in results]
        memory_usages = [r["memory_usage_bytes"] for r in results]
        
        # Calculate milicpu-seconds
        # milicpu-seconds = (CPU usage % / 100) * execution_time * 1000
        milicpu_seconds = [
            (cpu / 100.0) * time * 1000
            for cpu, time in zip(cpu_usages, inference_times)
        ]

        # Calculate miligpu-seconds (for GPU workloads)
        # Use container GPU utilizations if available, otherwise fall back to host stats
        miligpu_seconds = None
        gpu_utilizations_for_calc = []
        
        # Try to get GPU utilizations from container logs first
        container_gpu_utils = [r.get("gpu_utilization") for r in results if r.get("gpu_utilization") is not None]
        if container_gpu_utils:
            gpu_utilizations_for_calc = container_gpu_utils
        else:
            # Fall back to host GPU stats
            gpu_stats_list = [r["gpu_stats"] for r in results if r.get("gpu_stats")]
            if gpu_stats_list:
                gpu_utilizations_for_calc = [g["gpu_utilization"] for g in gpu_stats_list]
        
        if gpu_utilizations_for_calc:
            miligpu_seconds = [
                (gpu / 100.0) * time * 1000
                for gpu, time in zip(gpu_utilizations_for_calc, inference_times[:len(gpu_utilizations_for_calc)])
            ]

        # Calculate TOPS (Tera Operations Per Second)
        # TOPS = (Model Parameters * 2) / Inference Time (seconds) / 1e12
        # Factor of 2 accounts for multiply-accumulate operations (MACs)
        tops_values = []
        for inf_time in inference_times:
            if inf_time > 0:
                # Estimate operations: 2 * params per forward pass
                operations = self.model_params * 2
                tops = (operations / inf_time) / 1e12
                tops_values.append(tops)

        benchmark_results = {
            "workload_type": self.workload_manager.__class__.__name__,
            "model_path": self.model_loader.get_model_path(),
            "model_size_mb": self.model_loader.get_model_size() / 1024 / 1024,
            "model_params": self.model_params,
            "num_iterations": self.num_iterations,
            "inference_times": {
                "mean": statistics.mean(inference_times),
                "median": statistics.median(inference_times),
                "std": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                "min": min(inference_times),
                "max": max(inference_times),
                "all": inference_times,
            },
            "cpu_usage": {
                "mean": statistics.mean(cpu_usages),
                "median": statistics.median(cpu_usages),
                "std": statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0,
                "min": min(cpu_usages),
                "max": max(cpu_usages),
            },
            "memory_usage": {
                "mean_mb": statistics.mean(memory_usages) / 1024 / 1024,
                "max_mb": max(memory_usages) / 1024 / 1024,
            },
            "milicpu_seconds": {
                "mean": statistics.mean(milicpu_seconds),
                "median": statistics.median(milicpu_seconds),
                "total": sum(milicpu_seconds),
            },
            "tops": {
                "mean": statistics.mean(tops_values) if tops_values else 0,
                "median": statistics.median(tops_values) if tops_values else 0,
                "std": statistics.stdev(tops_values) if len(tops_values) > 1 else 0,
                "min": min(tops_values) if tops_values else 0,
                "max": max(tops_values) if tops_values else 0,
                "all": tops_values,
            },
        }

        # Add GPU information
        gpu_used = results[0].get("gpu_used", False) if results else False
        benchmark_results["gpu_used"] = gpu_used
        
        if miligpu_seconds:
            benchmark_results["miligpu_seconds"] = {
                "mean": statistics.mean(miligpu_seconds),
                "median": statistics.median(miligpu_seconds),
                "total": sum(miligpu_seconds),
            }
        
        # Add GPU usage stats (prefer container stats, fall back to host)
        if container_gpu_utils:
            benchmark_results["gpu_usage"] = {
                "mean": statistics.mean(container_gpu_utils),
                "median": statistics.median(container_gpu_utils),
                "std": statistics.stdev(container_gpu_utils) if len(container_gpu_utils) > 1 else 0,
                "min": min(container_gpu_utils),
                "max": max(container_gpu_utils),
                "source": "container",
            }
        else:
            # Fall back to host GPU stats
            gpu_stats_list = [r["gpu_stats"] for r in results if r.get("gpu_stats")]
            if gpu_stats_list:
                benchmark_results["gpu_usage"] = {
                    "mean": statistics.mean([g["gpu_utilization"] for g in gpu_stats_list]),
                    "power_watts_mean": statistics.mean([g["power_watts"] for g in gpu_stats_list]),
                    "source": "host",
                }

        return benchmark_results

