.PHONY: help install build test clean docker-build docker-up docker-down benchmark-cpu benchmark-gpu benchmark-coral benchmark-coral-max benchmark-all benchmark-tiny-llm

help: ## Show this help message
	@echo "LLMs-FPGA Benchmark - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using UV sync
	uv sync

install-dev: ## Install dependencies with dev tools
	uv sync --dev

build: ## Build the project (sync dependencies)
	uv sync

docker-build: ## Build the Docker images (standard and max)
	docker build -t llms-fpga-benchmark:latest .
	docker build -f Dockerfile.max -t llms-fpga-benchmark-max:latest .

docker-up-cpu: ## Start CPU benchmark container
	docker-compose --profile cpu up -d cpu-benchmark

docker-up-gpu: ## Start GPU benchmark container
	docker-compose --profile gpu up -d gpu-benchmark

docker-up-coral: ## Start Coral benchmark container (standard driver)
	docker-compose --profile coral up -d coral-benchmark

docker-up-coral-max: ## Start Coral benchmark container (max performance driver)
	docker-compose --profile coral-max up -d coral-benchmark-max

docker-down: ## Stop all containers
	docker-compose down

docker-logs-cpu: ## View CPU container logs
	docker-compose logs -f cpu-benchmark

docker-logs-gpu: ## View GPU container logs
	docker-compose logs -f gpu-benchmark

docker-logs-coral: ## View Coral container logs (standard driver)
	docker-compose logs -f coral-benchmark

docker-logs-coral-max: ## View Coral container logs (max performance driver)
	docker-compose logs -f coral-benchmark-max

benchmark-cpu: ## Run CPU benchmark (requires MODEL_PATH env var)
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not set. Usage: make benchmark-cpu MODEL_PATH=models/your_model.tflite"; \
		exit 1; \
	fi
	@mkdir -p results
	uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload cpu --iterations 100 --use-prompts --output results/cpu_benchmark.json

benchmark-gpu: ## Run GPU benchmark (requires MODEL_PATH env var)
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not set. Usage: make benchmark-gpu MODEL_PATH=models/your_model.tflite"; \
		exit 1; \
	fi
	@mkdir -p results
	uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload gpu --iterations 100 --use-prompts --output results/gpu_benchmark.json

benchmark-coral: ## Run Coral benchmark (standard driver, requires MODEL_PATH env var)
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not set. Usage: make benchmark-coral MODEL_PATH=models/your_model.tflite"; \
		exit 1; \
	fi
	@mkdir -p results
	uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload coral --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_benchmark.json

benchmark-coral-max: ## Run Coral benchmark (max performance driver, requires MODEL_PATH env var)
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not set. Usage: make benchmark-coral-max MODEL_PATH=models/your_model.tflite"; \
		exit 1; \
	fi
	@mkdir -p results
	uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload coral-max --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_max_benchmark.json

benchmark-all: ## Run all benchmarks (CPU, GPU, Coral, Coral-Max) and save results to results/ folder
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not set. Usage: make benchmark-all MODEL_PATH=models/your_model.tflite"; \
		echo "Example: make benchmark-all MODEL_PATH=models/tiny_llm.tflite"; \
		exit 1; \
	fi
	@mkdir -p results
	@echo "========================================="
	@echo "Running CPU benchmark..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload cpu --iterations 100 --use-prompts --output results/cpu_benchmark.json || echo "CPU benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running GPU benchmark..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload gpu --iterations 100 --use-prompts --output results/gpu_benchmark.json || echo "GPU benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running Coral benchmark (standard driver)..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload coral --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_benchmark.json || echo "Coral benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running Coral benchmark (max performance driver)..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model $(MODEL_PATH) --workload coral-max --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_max_benchmark.json || echo "Coral-Max benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "All benchmarks completed!"
	@echo "Results saved to results/ folder:"
	@ls -lh results/*.json 2>/dev/null || echo "No results files found"
	@echo "========================================="

benchmark-tiny-llm: ## Download, convert Tiny-LLM, and run all benchmarks including Coral-Max (one-command solution)
	@mkdir -p results models
	@echo "========================================="
	@echo "Downloading and converting Tiny-LLM..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model models/tiny_llm.tflite --workload cpu --iterations 1 --convert-tiny-llm > /dev/null 2>&1 || true
	@echo ""
	@echo "========================================="
	@echo "Running CPU benchmark..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model models/tiny_llm.tflite --workload cpu --iterations 100 --use-prompts --output results/cpu_benchmark.json || echo "CPU benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running GPU benchmark..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model models/tiny_llm.tflite --workload gpu --iterations 100 --use-prompts --output results/gpu_benchmark.json || echo "GPU benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running Coral benchmark (standard driver)..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model models/tiny_llm.tflite --workload coral --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_benchmark.json || echo "Coral benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "Running Coral benchmark (max performance driver)..."
	@echo "========================================="
	@uv run python -m llms_fpga_benchmark.cli --model models/tiny_llm.tflite --workload coral-max --iterations 100 --use-prompts --coral-device /dev/apex_0 --output results/coral_max_benchmark.json || echo "Coral-Max benchmark failed"
	@echo ""
	@echo "========================================="
	@echo "All benchmarks completed!"
	@echo "Results saved to results/ folder:"
	@ls -lh results/*.json 2>/dev/null || echo "No results files found"
	@echo "========================================="

test: ## Run tests (when implemented)
	pytest

lint: ## Run linters
	ruff check src/
	black --check src/

format: ## Format code
	black src/
	ruff check --fix src/

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +

clean-docker: ## Clean Docker containers and images
	docker-compose down -v
	docker rmi llms-fpga-benchmark:latest || true

