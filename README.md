# LLMs-FPGA Benchmark

A comprehensive benchmarking tool for measuring speed and resource consumption of Transformer model inference on different hardware platforms: CPU, GPU (CUDA), and Google Coral (with both standard and max performance drivers). Supports automatic conversion of Hugging Face models (including Tiny-LLM) to TensorFlow Lite format for Coral Edge TPU.

## Features

- **Multi-platform Support**: Benchmark inference on CPU, GPU (CUDA), and Google Coral (standard and max performance drivers)
- **Resource Monitoring**: Track CPU usage, memory consumption, and GPU utilization (measured inside container)
- **GPU Detection**: Automatic GPU availability detection and usage flag logging (`GPU_USED=True/False`)
- **Docker-based**: Isolated execution environment with resource monitoring
- **Modular Design**: Clean separation of concerns with base classes and workload-specific implementations
- **Metrics**: Results in millicpu-seconds, milligpu-seconds, and TOPS (Tera Operations Per Second)
- **Model Conversion**: Automatic download and conversion of Hugging Face models (Tiny-LLM) to TFLite
- **Prompt Generation**: Built-in prompt generator with 10 test prompts for text generation benchmarks
- **Modern Tooling**: Uses UV for fast Python package management and pyproject.toml for dependency management

## Project Structure

```
LLMs-FPGA/
├── src/
│   └── llms_fpga_benchmark/
│       ├── __init__.py
│       ├── model_loader.py        # Model loading and validation
│       ├── model_converter.py     # Hugging Face model conversion
│       ├── prompt_generator.py    # Prompt generation for benchmarks
│       ├── docker_manager.py      # Base Docker management class
│       ├── workloads.py           # CPU, GPU, and Coral workload managers
│       ├── benchmark.py           # Benchmarking logic with TOPS calculation
│       └── cli.py                 # Command-line interface
├── docker/
│   └── run_inference.py           # Inference script (runs in container)
├── Dockerfile                     # Multi-platform Docker image
├── docker-compose.yml             # Docker Compose configuration
├── pyproject.toml                 # Project dependencies (UV)
└── README.md                      # This file
```

## Prerequisites

- **Linux** (Ubuntu/Debian recommended)
- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **NVIDIA Docker** (for GPU workloads): [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **UV** (Python package manager): Install from [uv documentation](https://github.com/astral-sh/uv) or `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   cd /home/desktop/projects/LLMs-FPGA
   ```

2. **Install dependencies using UV sync**:
   ```bash
   uv sync
   ```
   
   For development with dev dependencies:
   ```bash
   uv sync --dev
   ```

3. **Generate lock file** (recommended for reproducible builds):
   ```bash
   uv sync  # This will create uv.lock
   ```
   
   **Note**: For reproducible Docker builds, it's recommended to commit `uv.lock` to the repository. 
   If you want to commit it, remove `uv.lock` from `.gitignore`.

4. **Build the Docker images**:
   ```bash
   # Build standard image (for CPU, GPU, and Coral standard driver)
   docker build -t llms-fpga-benchmark:latest .
   
   # Build max performance image (for Coral max driver - 500 MHz)
   docker build -f Dockerfile.max -t llms-fpga-benchmark-max:latest .
   
   # Or use Makefile to build both:
   make docker-build
   ```

## Usage

### Quick Start

#### Option 1: Use Tiny-LLM (Automatic Conversion)

**Run benchmark with automatic Tiny-LLM conversion**:
```bash
# CPU benchmark with Tiny-LLM and prompts
uv run python -m llms_fpga_benchmark.cli \
  --model models/tiny_llm.tflite \
  --workload cpu \
  --iterations 100 \
  --convert-tiny-llm \
  --use-prompts

# Or using the entry point
uv run llms-fpga-benchmark \
  --model models/tiny_llm.tflite \
  --workload cpu \
  --iterations 100 \
  --convert-tiny-llm \
  --use-prompts

# Coral benchmark with Tiny-LLM (standard driver - 250 MHz)
uv run python -m llms_fpga_benchmark.cli \
  --model models/tiny_llm.tflite \
  --workload coral \
  --iterations 100 \
  --convert-tiny-llm \
  --use-prompts \
  --coral-device /dev/apex_0

# Coral benchmark with Tiny-LLM (max performance driver - 500 MHz)
uv run python -m llms_fpga_benchmark.cli \
  --model models/tiny_llm.tflite \
  --workload coral-max \
  --iterations 100 \
  --convert-tiny-llm \
  --use-prompts \
  --coral-device /dev/apex_0
```

#### Option 2: Use Your Own Model

1. **Prepare your model**: Place your TensorFlow Lite model file in the `models/` directory
   ```bash
   mkdir -p models
   cp your_model.tflite models/
   ```

2. **Run a benchmark**:
   ```bash
   # CPU benchmark
   uv run python -m llms_fpga_benchmark.cli --model models/your_model.tflite --workload cpu --iterations 100
   
   # Or using the entry point
   uv run llms-fpga-benchmark --model models/your_model.tflite --workload cpu --iterations 100

   # GPU benchmark
   uv run python -m llms_fpga_benchmark.cli --model models/your_model.tflite --workload gpu --iterations 100

   # Coral benchmark (standard driver)
   uv run python -m llms_fpga_benchmark.cli --model models/your_model.tflite --workload coral --iterations 100 --coral-device /dev/apex_0
   
   # Coral benchmark (max performance driver)
   uv run python -m llms_fpga_benchmark.cli --model models/your_model.tflite --workload coral-max --iterations 100 --coral-device /dev/apex_0
   ```

   **Using Makefile** (alternative):
   ```bash
   # CPU benchmark
   make benchmark-cpu MODEL_PATH=models/your_model.tflite
   
   # GPU benchmark
   make benchmark-gpu MODEL_PATH=models/your_model.tflite
   
   # Coral benchmark (standard driver - 250 MHz)
   make benchmark-coral MODEL_PATH=models/your_model.tflite
   
   # Coral benchmark (max performance driver - 500 MHz)
   make benchmark-coral-max MODEL_PATH=models/your_model.tflite
   ```

#### Option 3: Run All Benchmarks at Once

**One-command solution with Tiny-LLM** (downloads, converts, and runs all benchmarks):
```bash
make benchmark-tiny-llm
```
This will:
- Download and convert Tiny-LLM from Hugging Face (if not already present)
- Run CPU, GPU, Coral (standard), and Coral-Max benchmarks sequentially
- Save all results to `results/` folder as JSON files:
  - `results/cpu_benchmark.json`
  - `results/gpu_benchmark.json`
  - `results/coral_benchmark.json`
  - `results/coral_max_benchmark.json`

**Run all benchmarks with your own model**:
```bash
make benchmark-all MODEL_PATH=models/your_model.tflite
```
This runs CPU, GPU, Coral (standard), and Coral-Max benchmarks.

All Makefile benchmark targets:
- Automatically create the `results/` directory
- Save results to JSON files in `results/`
- Use prompts for text generation (`--use-prompts`)
- Run 100 iterations by default

### Using Makefile Commands

The Makefile provides convenient shortcuts for common operations:

```bash
# Show all available commands
make help

# Install dependencies
make install

# Build Docker image
make docker-build

# Run all benchmarks (Tiny-LLM - one command!)
make benchmark-tiny-llm

# Run all benchmarks with your model
make benchmark-all MODEL_PATH=models/your_model.tflite

# Run individual benchmarks
make benchmark-cpu MODEL_PATH=models/your_model.tflite
make benchmark-gpu MODEL_PATH=models/your_model.tflite
make benchmark-coral MODEL_PATH=models/your_model.tflite
make benchmark-coral-max MODEL_PATH=models/your_model.tflite
```

All benchmark commands automatically:
- Create the `results/` directory if it doesn't exist
- Save results as JSON files in `results/`
- Use prompts for text generation
- Run 100 iterations by default

### Using Docker Compose

Docker Compose provides an easy way to manage containers for different workloads:

```bash
# Start CPU container
docker-compose --profile cpu up -d cpu-benchmark

# Start GPU container
docker-compose --profile gpu up -d gpu-benchmark

# Start Coral container (standard driver)
docker-compose --profile coral up -d coral-benchmark

# Start Coral container (max performance driver)
docker-compose --profile coral-max up -d coral-benchmark-max
```

### Command-Line Options

```
--model PATH          Path to TensorFlow Lite model file (required)
--workload TYPE       Workload type: cpu, gpu, coral, or coral-max (required)
                      coral-max uses max performance driver (500 MHz)
--iterations N        Number of benchmark iterations (default: 100)
--warmup N            Number of warmup iterations (default: 2)
--output FILE         Output file for JSON results (default: stdout)
--coral-device PATH   Path to Coral device (e.g., /dev/apex_0)
--image NAME          Docker image name (default: llms-fpga-benchmark:latest)
--convert-tiny-llm    Automatically download and convert Tiny-LLM from Hugging Face
--use-prompts         Use generated prompts for text generation benchmarks
--model-params N      Number of model parameters for TOPS calculation (default: 10M)
```

### Output Format

The benchmark outputs JSON with the following structure:

```json
{
  "workload_type": "CPUWorkloadManager",
  "model_path": "/app/models/model.tflite",
  "model_size_mb": 12.5,
  "num_iterations": 100,
  "inference_times": {
    "mean": 0.123,
    "median": 0.120,
    "std": 0.010,
    "min": 0.110,
    "max": 0.145,
    "all": [0.110, 0.120, ...]
  },
  "cpu_usage": {
    "mean": 85.5,
    "median": 87.0,
    "std": 5.2,
    "min": 80.0,
    "max": 90.0
  },
  "memory_usage": {
    "mean_mb": 512.0,
    "max_mb": 550.0
  },
  "milicpu_seconds": {
    "mean": 105.0,
    "median": 104.0,
    "total": 1050.0
  },
  "miligpu_seconds": {
    "mean": 95.0,
    "median": 94.0,
    "total": 950.0
  },
  "gpu_used": false,
  "gpu_usage": {
    "mean": 0.0,
    "median": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0,
    "source": "container"
  },
  "tops": {
    "mean": 0.045,
    "median": 0.044,
    "std": 0.002,
    "min": 0.042,
    "max": 0.048,
    "all": [0.042, 0.044, ...]
  },
  "model_params": 10000000
}
```

## Architecture

### Base Classes

- **`DockerWorkloadManager`**: Abstract base class for managing Docker containers
  - Handles container lifecycle (start/stop)
  - Provides methods for executing commands and collecting stats
  - Context manager support for clean resource management

### Workload Implementations

- **`CPUWorkloadManager`**: CPU-only inference workloads
- **`GPUWorkloadManager`**: GPU (CUDA) inference workloads with NVIDIA support
- **`CoralWorkloadManager`**: Google Coral inference workloads with USB device access (standard driver - 250 MHz)
- **`CoralMaxWorkloadManager`**: Google Coral inference workloads with USB device access (max performance driver - 500 MHz)

### Model Conversion

- **`ModelConverter`**: Downloads and converts Hugging Face models to TFLite
  - Supports Tiny-LLM from `arnir0/Tiny-LLM`
  - Quantizes models for Coral Edge TPU compatibility
  - Handles tokenizer preservation for text generation

### Prompt Generation

- **`PromptGenerator`**: Generates test prompts for benchmarking
  - Includes 10 default prompts for text generation
  - Supports custom prompt lists
  - Rotates prompts across benchmark iterations

### Benchmarking

The `BenchmarkRunner` class:
- Runs warmup iterations to stabilize performance
- Executes multiple benchmark iterations in batch mode (model loaded once for efficiency)
- Collects resource usage statistics (CPU, memory, GPU) from both host and container
- Measures GPU load inside the container using `nvidia-smi` (for GPU workloads)
- Detects and logs GPU usage status (`GPU_USED=True/False`)
- Calculates millicpu-seconds, milligpu-seconds, and TOPS metrics
- Provides statistical summaries (mean, median, std, min, max)
- TOPS calculation: `(Model Parameters × 2) / Inference Time / 1e12`

## Tiny-LLM Model

The benchmark supports automatic download and conversion of [Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM) from Hugging Face:

- **Model**: `arnir0/Tiny-LLM` (10M parameters)
- **Automatic Conversion**: Downloads and converts to TFLite format
- **Quantization**: Automatically quantizes for Coral Edge TPU compatibility
- **Tokenizer**: Preserves tokenizer for text generation

The model is cached locally to avoid re-downloading on subsequent runs.

## Docker Images

The project uses two Docker images:

1. **`llms-fpga-benchmark:latest`** (from `Dockerfile`):
   - Python 3.9 (compatible with Coral requirements)
   - **Edge TPU Runtime Library** (`libedgetpu1-std`) - standard performance (250 MHz)
   - Used for CPU, GPU, and Coral (standard) workloads

2. **`llms-fpga-benchmark-max:latest`** (from `Dockerfile.max`):
   - Python 3.9 (compatible with Coral requirements)
   - **Edge TPU Runtime Library** (`libedgetpu1-max`) - maximum performance (500 MHz)
   - Used for Coral-Max workloads
   - **Warning**: Max driver can run hot - use with adequate cooling

Both images include:
- **Coral-specific TensorFlow Lite runtime** - installed from local wheel files (if available in `whls/`) or Debian packages
- **PyCoral API** - installed from local wheel files (if available in `whls/`) or Debian packages for Edge TPU support
- Transformers and PyTorch (for model conversion)
- CUDA support (via host NVIDIA drivers)
- All project dependencies

### Coral Dependencies Installation

The Dockerfiles install Coral dependencies in two parts:

1. **Edge TPU Runtime Library**:
   - `Dockerfile`: Installs `libedgetpu1-std` (250 MHz - standard performance)
   - `Dockerfile.max`: Installs `libedgetpu1-max` (500 MHz - maximum performance, can run hot)

2. **Python Dependencies**: Supports two methods:
   - **Local Wheel Files (Recommended)**: Place wheel files in the `whls/` directory:
     - `pycoral-*.whl` (e.g., `pycoral-2.0.0-cp39-cp39-linux_x86_64.whl`)
     - `tflite_runtime-*.whl` (e.g., `tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl`)
   - **Debian Packages (Fallback)**: If wheel files are not found, installs `python3-tflite-runtime` and `python3-pycoral` from the Coral Debian repository.

**Note**: The Coral-specific TensorFlow Lite runtime is different from the standard `tensorflow-lite-runtime` package. Using local wheel files gives you more control over versions and ensures compatibility with your specific setup.

## Google Coral Setup

For Google Coral workloads, you can choose between two performance modes:

### Standard Driver (250 MHz)
- Lower power consumption
- More stable temperature
- Recommended for continuous operation

### Max Performance Driver (500 MHz)
- Maximum performance
- Higher power consumption
- Can run hot - ensure adequate cooling
- Recommended for performance-critical benchmarks

### Setup Steps

1. **Connect the Coral device** via USB
2. **Identify the device path**:
   ```bash
   lsusb | grep -i coral
   ls -la /dev/apex_*
   ```
3. **Run benchmark** with the device path:
   ```bash
   # Standard driver (250 MHz)
   uv run python -m llms_fpga_benchmark.cli \
     --model models/model.tflite \
     --workload coral \
     --coral-device /dev/apex_0
   
   # Max performance driver (500 MHz)
   uv run python -m llms_fpga_benchmark.cli \
     --model models/model.tflite \
     --workload coral-max \
     --coral-device /dev/apex_0
   ```

**Note**: The max driver uses `Dockerfile.max` which installs `libedgetpu1-max`. Make sure to build the max image:
```bash
docker build -f Dockerfile.max -t llms-fpga-benchmark-max:latest .
# Or use: make docker-build
```

## GPU Setup

For GPU workloads:

1. **Install NVIDIA drivers** on the host
2. **Install nvidia-container-toolkit**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### GPU Monitoring

The benchmark automatically measures GPU load inside the container:
- **GPU Detection**: Checks if GPU is available using `nvidia-smi`
- **GPU Usage Flag**: Logs `GPU_USED=True/False` to indicate if TensorFlow is using the GPU
- **GPU Metrics**: Measures GPU utilization, memory usage, and power consumption from inside the container
- **Process-Level Monitoring**: Tracks GPU usage per process using `nvidia-smi` and `pynvml`
- **Results**: GPU statistics are included in the JSON output with a `source` field indicating whether metrics came from the container or host

**Note**: 
- For GPU workloads, the benchmark uses TensorFlow (not TFLite) when a SavedModel is available, which enables GPU acceleration
- The SavedModel is automatically created during model conversion (when using `--convert-tiny-llm`)
- If `GPU_USED=True`, TensorFlow is using GPU acceleration
- If `GPU_USED=False`, inference is running on CPU (either TFLite or TensorFlow without GPU)

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
uv sync --dev

# Run tests (when implemented)
uv run pytest

# Format code
uv run black src/

# Lint code
uv run ruff check src/
```

### Adding New Workload Types

1. Create a new class inheriting from `DockerWorkloadManager`
2. Implement `get_container_config()` method
3. Add the new workload type to `workloads.py`
4. Update CLI to support the new workload type

## Limitations

- **Linux only**: Currently designed for Linux systems
- **Coral**: Single-task execution only (no consumption metrics)
- **Model format**: Currently supports TensorFlow Lite models (and TensorFlow SavedModel for GPU)
- **Python version**: Python 3.9 required (Coral compatibility)
- **Coral Max Driver**: Can run hot - ensure adequate cooling when using `coral-max` workload

## Troubleshooting

### Container fails to start
- Check Docker is running: `docker ps`
- Verify image exists: `docker images | grep llms-fpga-benchmark`
- Check logs: `docker logs <container-name>`

### GPU not detected
- Verify NVIDIA drivers: `nvidia-smi`
- Check nvidia-container-toolkit: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Ensure Docker daemon has GPU support enabled
- Check logs for `GPU_AVAILABLE=True/False` and `GPU_USED=True/False` flags
- Note: `GPU_USED=False` is normal for standard TensorFlow Lite (it uses CPU by default)

### Coral device not found
- Check USB connection: `lsusb`
- Verify device permissions: `ls -la /dev/apex_*`
- Try running with `--privileged` flag or adjust device path

## License

[Specify your license here]

## Contributing

[Contributing guidelines if applicable]

