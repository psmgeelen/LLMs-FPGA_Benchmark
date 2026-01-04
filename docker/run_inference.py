"""Inference script that runs inside the Docker container."""

import argparse
import sys
import time
import logging
import subprocess
import os
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_available() -> bool:
    """Check if GPU is available and accessible."""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, timeout=2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def get_gpu_stats() -> Optional[Dict[str, Any]]:
    """Get GPU statistics from nvidia-smi (system-wide)."""
    try:
        # Get GPU utilization
        util_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', 
                                    '--format=csv,noheader,nounits'], 
                                   capture_output=True, timeout=2, text=True)
        if util_result.returncode == 0:
            parts = util_result.stdout.strip().split(', ')
            if len(parts) >= 5:
                return {
                    "gpu_utilization": float(parts[0]),
                    "memory_utilization": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "power_watts": float(parts[4]),
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception) as e:
        logger.debug(f"Could not get GPU stats: {e}")
    return None

def get_process_gpu_stats(pid: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get GPU statistics for a specific process using nvidia-smi.
    
    Args:
        pid: Process ID to monitor. If None, uses current process.
    
    Returns:
        Dictionary with process-specific GPU stats, or None if unavailable
    """
    if pid is None:
        pid = os.getpid()
    
    try:
        # Get process GPU stats using nvidia-smi
        # Query processes and filter by PID
        query = f'--query-compute-apps=pid,process_name,used_memory,sm_util --format=csv,noheader,nounits'
        result = subprocess.run(['nvidia-smi', query], 
                              capture_output=True, timeout=2, text=True)
        
        if result.returncode == 0:
            # Parse output to find our process
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            proc_pid = int(parts[0])
                            if proc_pid == pid:
                                return {
                                    "pid": proc_pid,
                                    "process_name": parts[1],
                                    "gpu_memory_used_mb": float(parts[2]),
                                    "sm_utilization": float(parts[3]),  # Streaming multiprocessor utilization
                                }
                        except (ValueError, IndexError):
                            continue
        
        # Fallback: Try to get process info from nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get running processes
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                if proc.pid == pid:
                    # Get utilization for this process (approximate)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    return {
                        "pid": pid,
                        "gpu_memory_used_mb": proc.usedGpuMemory / 1024 / 1024,
                        "gpu_utilization": util.gpu,  # System-wide, but process is contributing
                        "sm_utilization": util.gpu,  # Approximate
                    }
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get process GPU stats via pynvml: {e}")
            
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception) as e:
        logger.debug(f"Could not get process GPU stats: {e}")
    return None

def detect_tflite_gpu_usage(interpreter, initial_gpu_stats: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect if TensorFlow Lite interpreter is actually using GPU.
    
    This checks multiple indicators:
    1. If interpreter has GPU delegates loaded
    2. If GPU utilization increases during inference (more reliable)
    
    Args:
        interpreter: TFLite interpreter instance
        initial_gpu_stats: Initial GPU stats before inference (for comparison)
    
    Returns:
        True if GPU is actually being used, False otherwise
    """
    try:
        # Method 1: Check if interpreter has GPU delegate loaded
        if hasattr(interpreter, '_delegates') and interpreter._delegates:
            logger.debug("GPU delegate found in interpreter._delegates")
            return True
        
        # Method 2: Check interpreter's internal delegate list
        try:
            # Some TFLite versions expose delegates differently
            if hasattr(interpreter, 'get_delegates'):
                delegates = interpreter.get_delegates()
                if delegates:
                    logger.debug(f"GPU delegates found via get_delegates(): {delegates}")
                    return True
        except Exception:
            pass
        
        # Method 3: Check TensorFlow operations (if using full TensorFlow)
        # Standard TFLite runtime doesn't use GPU, so this will be False
        # Only TensorFlow Lite with GPU delegate would return True
        
        # Default: Standard TFLite doesn't use GPU by default
        # This is the expected behavior - standard TFLite runs on CPU
        return False
    except Exception as e:
        logger.debug(f"Error detecting GPU usage: {e}")
        return False

def run_inference(
    model_path: str,
    iteration: int = 0,
    use_coral: bool = False,
    prompt: Optional[str] = None,
    max_length: int = 50,
    num_iterations: int = 1,
    prompts: Optional[list] = None,
):
    """
    Run inference on a TensorFlow Lite model.
    
    Supports both standard TFLite and Coral Edge TPU inference.
    Can perform text generation if a prompt is provided.
    Supports batch mode: load model once, run multiple iterations.
    
    Args:
        model_path: Path to the TFLite model file
        iteration: Iteration number for logging (starting offset)
        use_coral: Whether to use Coral Edge TPU (default: False)
        prompt: Optional text prompt for generation (single iteration)
        max_length: Maximum generation length (default: 50)
        num_iterations: Number of iterations to run (batch mode, default: 1)
        prompts: Optional list of prompts for batch mode
    """
    try:
        import numpy as np
        
        # Load tokenizer once if prompts are provided
        tokenizer = None
        tokenizer_path = model_path.replace('.tflite', '_tokenizer')
        if prompt or (prompts and len(prompts) > 0):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
        
        # Try Coral first if requested
        if use_coral:
            try:
                from pycoral.utils import edgetpu
                from pycoral.utils import dataset
                from pycoral.adapters import common
                from pycoral.adapters import classify
                
                # Load model once for all iterations
                logger.info(f"Loading model on Coral Edge TPU: {model_path}")
                interpreter = edgetpu.make_interpreter(model_path)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                input_shape = input_details[0]['shape']
                
                # Run batch of iterations
                for i in range(num_iterations):
                    current_iteration = iteration + i
                    current_prompt = None
                    
                    if prompts and i < len(prompts):
                        current_prompt = prompts[i]
                    elif prompt and i == 0:
                        current_prompt = prompt
                    
                    # Prepare input
                    if current_prompt and tokenizer:
                        seq_length = input_shape[1] if len(input_shape) > 1 else 128
                        encoded = tokenizer.encode(current_prompt, max_length=seq_length, truncation=True, padding="max_length", return_tensors="np")
                        input_data = encoded.astype(np.uint8) % 256
                        logger.info(f"[Iteration {current_iteration}] Using prompt: {current_prompt[:50]}...")
                    else:
                        input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
                    
                    # Run inference
                    start_time = time.time()
                    common.set_input(interpreter, input_data)
                    interpreter.invoke()
                    inference_time = time.time() - start_time
                    
                    # Get output
                    output_data = classify.get_scores(interpreter)
                    
                    logger.info(f"[Iteration {current_iteration}] Coral inference completed in {inference_time:.4f}s")
                
                return True
                
            except ImportError:
                logger.warning("PyCoral not available, falling back to standard TFLite")
                use_coral = False
            except Exception as e:
                logger.warning(f"Coral inference failed: {e}, falling back to standard TFLite")
                use_coral = False
        
        # Standard TensorFlow Lite inference OR TensorFlow (for GPU)
        if not use_coral:
            # Check GPU availability
            gpu_available = check_gpu_available()
            logger.info(f"GPU_AVAILABLE={gpu_available}")
            
            # For GPU workloads, try TensorFlow first (better GPU support)
            # For CPU workloads, use TFLite (more efficient)
            use_tensorflow = gpu_available and os.environ.get('USE_TENSORFLOW_GPU', 'false').lower() == 'true'
            
            # Try to find SavedModel version if TFLite is provided
            savedmodel_path = None
            if model_path.endswith('.tflite') and use_tensorflow:
                # Look for corresponding SavedModel directory
                # SavedModel is saved as a directory, not a file
                base_path = model_path.replace('.tflite', '')
                savedmodel_path = f"{base_path}_savedmodel"
                logger.info(f"Looking for SavedModel at: {savedmodel_path}")
                if not os.path.exists(savedmodel_path) or not os.path.isdir(savedmodel_path):
                    logger.warning(f"SavedModel not found at {savedmodel_path}, falling back to TFLite")
                    savedmodel_path = None
                else:
                    logger.info(f"Found SavedModel at: {savedmodel_path}")
            
            if use_tensorflow and savedmodel_path:
                # Use TensorFlow for GPU acceleration
                try:
                    import tensorflow as tf
                    import numpy as np
                    
                    # Enable GPU memory growth to avoid allocating all memory
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            logger.info(f"TensorFlow GPU devices available: {len(gpus)}")
                            using_gpu = True
                        except RuntimeError as e:
                            logger.warning(f"Could not configure GPU memory growth: {e}")
                            using_gpu = False
                    else:
                        using_gpu = False
                    
                    # Load SavedModel
                    logger.info(f"Loading TensorFlow SavedModel for GPU acceleration: {savedmodel_path}")
                    model = tf.saved_model.load(savedmodel_path)
                    
                    # Get model signature
                    if hasattr(model, 'signatures'):
                        infer = model.signatures['serving_default']
                    else:
                        # Fallback: assume model is callable
                        infer = model
                    
                    logger.info("TensorFlow SavedModel loaded successfully")
                    logger.info(f"GPU_USED={using_gpu}")
                    
                    # Get input shape from model signature
                    try:
                        if hasattr(infer, 'structured_input_signature'):
                            # Try to get input shape from signature
                            sig = infer.structured_input_signature
                            if sig and len(sig) > 0:
                                input_shape = sig[0][0].shape.as_list()
                                if input_shape[0] is None:  # Batch dimension
                                    input_shape[0] = 1
                            else:
                                input_shape = (1, 192)  # Default
                        else:
                            input_shape = (1, 192)  # Default
                    except Exception:
                        input_shape = (1, 192)  # Default fallback
                    
                    input_dtype = np.float32
                    
                except ImportError:
                    logger.warning("TensorFlow not available, falling back to TFLite")
                    use_tensorflow = False
                except Exception as e:
                    logger.warning(f"Failed to load TensorFlow SavedModel: {e}, falling back to TFLite")
                    use_tensorflow = False
            else:
                use_tensorflow = False
            
            if not use_tensorflow:
                # Use TensorFlow Lite (CPU or TFLite GPU delegate)
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    try:
                        import tensorflow.lite as tflite
                    except ImportError:
                        logger.error("TensorFlow Lite not available")
                        return False

                # Load the model once for all iterations
                logger.info(f"Loading model with TensorFlow Lite: {model_path}")
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()

            if use_tensorflow:
                # TensorFlow model already loaded above
                pass
            else:
                # Get input and output details for TFLite
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                
                logger.info(f"Model loaded. Input shape: {input_shape}, Output shape: {output_details[0]['shape']}")
                
                # Initial GPU detection (before inference) - check for GPU delegate
                using_gpu = detect_tflite_gpu_usage(interpreter, None)
                logger.info(f"GPU_USED_INITIAL={using_gpu}")
            
            # Get initial GPU stats if available (for comparison)
            initial_gpu_stats = get_gpu_stats() if gpu_available else None
            initial_process_gpu_stats = get_process_gpu_stats() if gpu_available else None
            
            # Log process ID for monitoring
            current_pid = os.getpid()
            logger.info(f"PROCESS_PID={current_pid}")

            # Run batch of iterations
            for i in range(num_iterations):
                current_iteration = iteration + i
                current_prompt = None
                
                if prompts and i < len(prompts):
                    current_prompt = prompts[i]
                elif prompt and i == 0:
                    current_prompt = prompt
                
                # Prepare input - use prompt if provided, otherwise random
                if current_prompt and tokenizer:
                    seq_length = input_shape[1] if len(input_shape) > 1 else 192
                    encoded = tokenizer.encode(current_prompt, max_length=seq_length, truncation=True, padding="max_length", return_tensors="np")
                    
                    # Convert to appropriate dtype
                    if use_tensorflow:
                        input_data = encoded.astype(np.float32)
                    else:
                        # Convert to appropriate dtype for quantized model
                        if input_dtype == np.int8:
                            input_data = (encoded % 127).astype(np.int8)
                        elif input_dtype == np.int32:
                            input_data = encoded.astype(np.int32)
                        elif input_dtype == np.float32:
                            input_data = encoded.astype(np.float32)
                        else:
                            input_data = encoded.astype(input_dtype)
                    
                    logger.info(f"[Iteration {current_iteration}] Using prompt: {current_prompt[:50]}...")
                else:
                    # Generate appropriate input
                    if use_tensorflow:
                        input_data = np.random.randn(*input_shape).astype(np.float32)
                    else:
                        # Generate appropriate input based on dtype for 8-bit quantized models
                        if input_dtype == np.int8:
                            input_data = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
                        elif input_dtype == np.int32:
                            input_data = np.random.randint(0, 1000, size=input_shape, dtype=np.int32)
                        elif input_dtype == np.uint8:
                            input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
                        elif input_dtype == np.float32:
                            input_data = np.random.randn(*input_shape).astype(np.float32)
                        else:
                            input_data = np.random.randn(*input_shape).astype(input_dtype)
                
                # Run inference
                start_time = time.time()
                if use_tensorflow:
                    # Use TensorFlow model
                    input_tensor = tf.constant(input_data)
                    output_data = infer(input_tensor)
                    # Extract output from dict if needed
                    if isinstance(output_data, dict):
                        output_data = list(output_data.values())[0]
                    output_data = output_data.numpy()
                else:
                    # Use TFLite interpreter
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                
                inference_time = time.time() - start_time

                # Get GPU stats after inference (system-wide and process-level)
                gpu_stats = get_gpu_stats() if gpu_available else None
                process_gpu_stats = get_process_gpu_stats() if gpu_available else None
                
                # Re-detect GPU usage after first inference (check if utilization increased)
                # This is more reliable than pre-inference detection
                if i == 0:
                    if gpu_stats and initial_gpu_stats:
                        # After first inference, check if GPU utilization increased significantly
                        util_increase = gpu_stats['gpu_utilization'] - initial_gpu_stats.get('gpu_utilization', 0)
                        # Also check memory utilization increase as secondary indicator
                        mem_increase = gpu_stats['memory_utilization'] - initial_gpu_stats.get('memory_utilization', 0)
                        
                        # Threshold: if utilization increased by more than 5%, likely GPU usage
                        if util_increase > 5.0 or mem_increase > 5.0:
                            using_gpu = True
                            logger.info(f"GPU_USED_DETECTED=True (GPU util: +{util_increase:.1f}%, Mem util: +{mem_increase:.1f}%)")
                        else:
                            # Log that we checked but didn't detect GPU usage
                            logger.debug(f"GPU usage check: util change={util_increase:.1f}%, mem change={mem_increase:.1f}% (threshold: 5%)")
                    
                    # Check process-level GPU usage
                    if process_gpu_stats:
                        logger.info(f"PROCESS_GPU_MEM={process_gpu_stats.get('gpu_memory_used_mb', 0):.1f}MB PROCESS_GPU_UTIL={process_gpu_stats.get('sm_utilization', 0):.1f}%")
                
                # Build GPU info string
                gpu_info = ""
                if gpu_stats:
                    gpu_info = f" GPU_UTIL={gpu_stats['gpu_utilization']:.1f}% GPU_MEM={gpu_stats['memory_utilization']:.1f}%"
                if process_gpu_stats:
                    gpu_info += f" PROC_GPU_MEM={process_gpu_stats.get('gpu_memory_used_mb', 0):.1f}MB PROC_GPU_UTIL={process_gpu_stats.get('sm_utilization', 0):.1f}%"
                
                # Log final GPU_USED status (will be updated after first iteration if detected)
                logger.info(f"[Iteration {current_iteration}] Inference completed in {inference_time:.4f}s GPU_USED={using_gpu}{gpu_info}")
                logger.info(f"[Iteration {current_iteration}] Output shape: {output_data.shape}")
                
                # If we have a tokenizer and prompt, try to decode output
                if current_prompt and tokenizer:
                    try:
                        # Get top token from output
                        output_tokens = np.argmax(output_data, axis=-1) if len(output_data.shape) > 1 else np.argmax(output_data)
                        if isinstance(output_tokens, np.ndarray) and output_tokens.size > 0:
                            generated_text = tokenizer.decode(output_tokens.flatten()[:max_length], skip_special_tokens=True)
                            logger.info(f"[Iteration {current_iteration}] Generated: {generated_text[:100]}...")
                    except Exception as e:
                        logger.debug(f"Could not decode output: {e}")

        return True

    except Exception as e:
        logger.error(f"[Iteration {iteration}] Inference failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(description="Run inference on TFLite model")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model file")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (starting offset)")
    parser.add_argument("--coral", action="store_true", help="Use Coral Edge TPU")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation (single iteration)")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated prompts for batch mode")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run (batch mode, model loaded once)")
    
    args = parser.parse_args()
    
    # Parse prompts if provided
    prompts_list = None
    if args.prompts:
        prompts_list = [p.strip() for p in args.prompts.split(',')]
    
    success = run_inference(
        args.model,
        args.iteration,
        use_coral=args.coral,
        prompt=args.prompt,
        max_length=args.max_length,
        num_iterations=args.iterations,
        prompts=prompts_list,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

