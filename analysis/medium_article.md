# When GPUs Lose: Why Small LLMs Favor CPUs and Edge TPUs Over High-End GPUs

**Pieter Geelen**

The conventional wisdom in machine learning is clear: GPUs are faster. NVIDIA's CUDA ecosystem has become the de facto standard for training and inference, with frameworks optimized for GPU acceleration. But what happens when you challenge this assumption with a controlled, reproducible benchmark?

We benchmarked a 10-million parameter Transformer model across CPU, GPU, and Google Coral Edge TPU. The RTX 3090, despite its massive computational power, was 67% slower than a CPU for this workload. The Edge TPU matched CPU performance while consuming 85x less power.

This raises an important question: if specialized edge devices like the Coral TPU can match or exceed GPU performance for small models, how much more beneficial could a dedicated FPGA be? FPGAs offer programmability and the ability to create custom compute architectures optimized for specific workloads. Could they bridge the gap between general-purpose CPUs and specialized ASICs?

## The Problem: Hardware Assumptions in LLM Inference

Large Language Models have exploded in popularity, but not every use case requires a 70-billion parameter model running on a data center GPU. Edge deployment, real-time applications, and resource-constrained environments demand smaller models that can run efficiently on diverse hardware.

Yet the industry has largely converged on GPU-first solutions, driven by software availability and ecosystem momentum rather than fundamental hardware superiority. This creates a self-affirming cycle: CUDA is available, so frameworks optimize for CUDA, so developers use CUDA, so more frameworks optimize for CUDA.

But is this always the right choice?

## The Experiment: A Controlled Benchmark

To answer this question, we designed a rigorous benchmark comparing inference performance across four hardware platforms:

- **CPU**: AMD Ryzen 9 9950X (16 cores, 32 threads, 64MB L3 cache, 170W TDP)
- **GPU**: NVIDIA RTX 3090 (10,496 CUDA cores, 24GB GDDR6X, ~118W measured)
- **Coral Edge TPU (Standard)**: 250 MHz clock, ~2W average power
- **Coral Edge TPU (Max)**: 500 MHz clock, ~2.5W average power

The benchmark used a 10-million parameter Transformer model (Tiny-LLM), quantized to 8-bit integers for edge compatibility. This model size (~9.5MB) is intentionally small — small enough to fit entirely within the CPU's 64MB L3 cache, but representative of edge AI workloads.

## Methodology: Reproducibility Through Containerization

To ensure fair comparison and reproducible results, we containerized everything using Docker and the NVIDIA Container Toolkit. Each workload runs in an isolated container with identical software stacks, eliminating host system variations.

The Docker setup provides isolated environments where each workload type (CPU, GPU, Coral) runs with identical Python dependencies. Resource monitoring — CPU usage, memory consumption, and GPU utilization — is measured from inside the container, ensuring accurate per-workload metrics.

The NVIDIA Container Toolkit enables GPU access without host driver conflicts, while maintaining hardware abstraction. The same Docker image can be run on any compatible system, ensuring consistent results across different machines.

For GPU workloads, we use TensorFlow's SavedModel format (automatically generated during model conversion) rather than TensorFlow Lite, enabling proper GPU acceleration. The container detects GPU availability and automatically selects the appropriate backend, logging GPU usage status for verification.

This methodology ensures that we're comparing hardware capabilities, not software stack differences. By controlling for variables — identical models, same quantization (8-bit), same workload characteristics — we can isolate the impact of hardware architecture.

## The Results: When Cache Beats Compute

The benchmark results reveal a counterintuitive finding: the CPU outperforms the GPU for this small model.

**Inference Times (mean)**:
- CPU: 33.26 ms
- Coral (Max): 32.15 ms  
- Coral (Std): 33.23 ms
- GPU: 55.41 ms

The RTX 3090, despite its massive computational power, is 67% slower than the CPU. Why?

### The Cache Advantage

The 9.5MB model fits entirely within the CPU's 64MB L3 cache. This means L3 cache latency is ~40-75 cycles (~10-20ns), compared to GPU global memory latency of ~400-600 cycles (~100-150ns). PCIe transfer latency adds another ~5-10µs per transfer.

For small models, the CPU's cache hierarchy provides dramatically lower latency than GPU memory access. The GPU's parallel processing advantage is negated by the overhead of data transfers and kernel launches.

### The Overhead Problem

GPU inference involves significant overhead: kernel launch latency (~5-10µs even for small kernels), PCIe transfers between CPU and GPU memory, synchronization waiting for GPU operations to complete, and low utilization. Our benchmark showed only 8% GPU utilization, indicating the workload is too small to fully utilize GPU parallelism.

TensorFlow's Python API adds additional overhead: graph construction, session management, and memory allocation all contribute to the latency penalty. The framework overhead dominates for small workloads, making the GPU's computational advantage irrelevant.

## Edge TPUs: Specialized Hardware for Specialized Workloads

The Google Coral Edge TPU results are particularly interesting. At ~2W average power consumption, the Coral achieves performance comparable to a 170W CPU. The max performance driver (500 MHz) provides a slight edge, demonstrating that specialized hardware can excel when workloads match their design constraints.

**Power Efficiency (TOPS per Watt)**:
- Coral (Std): 0.000303 TOPS/W
- Coral (Max): 0.000249 TOPS/W
- CPU: 0.000004 TOPS/W
- GPU: 0.000003 TOPS/W

The Coral is 75-100x more power-efficient than the CPU or GPU for this workload. This isn't surprising — the Edge TPU is purpose-built for quantized neural network inference, with dedicated hardware for matrix operations. When the workload matches the hardware design, specialization wins.

## Hardware Development Lags Behind Software

This benchmark highlights a fundamental tension in computing: hardware development always lags behind software development.

Software frameworks evolve rapidly, driven by developer needs and ecosystem momentum. CUDA became dominant not because it was fundamentally superior for all workloads, but because NVIDIA made it available, well-documented, and integrated into popular frameworks. Once the ecosystem converged on CUDA, alternatives became harder to justify despite potential advantages.

We see this pattern throughout computing history. In networking, ASICs and FPGAs have become essential for achieving the efficiency needed at scale. Routers and switches use specialized hardware because general-purpose CPUs can't match the performance-per-watt requirements. The same principles apply to AI inference: when workloads are well-defined and performance-per-watt matters, specialized hardware wins.

It's no wonder that Google's TPUs are extremely interesting — they operate in a similar space. TPUs are custom ASICs designed specifically for machine learning workloads, offering higher performance-per-watt than GPUs for many ML operations. They're not general-purpose, but they excel at the specific operations common in neural networks.


## The Beginning of the End for CUDA Dominance?

CUDA and NVIDIA's ecosystem dominance might represent the beginning of an end, not the end itself. The current dominance is driven by availability — CUDA is widely available and well-supported. Software stacks are optimized for CUDA, most ML engineers know CUDA, and the ecosystem momentum creates a self-reinforcing cycle.

But in the long term, alternatives will become more interesting. Specialized hardware is maturing: Edge TPUs, NPUs, and custom ASICs are improving. Power efficiency matters more as edge deployment and mobile applications demand efficiency. As ML operations standardize, specialized hardware becomes more viable. And software abstraction is improving — frameworks that abstract hardware details enable easier migration.

The transition won't be immediate, but the trend is clear: when performance-per-watt and latency matter, specialized hardware wins. Availability of hardware and software stacks lead us to use NVIDIA products today, but long-term alternatives will become more compelling.

## Next Steps: NPUs and Dedicated AI Accelerators

In the next article, we'll explore Neural Processing Units (NPUs) like those found in the Rockchip RK3588, and dedicated AI accelerators like the Hailo-10H. These represent different approaches to edge AI acceleration.

NPUs like the RK3588's NPU represent a hybrid approach: they combine ASIC-style specialized compute units with general-purpose resources like RAM and CPU interfaces. This hybrid architecture offers specialized compute through dedicated hardware for neural network operations, while maintaining flexibility with general-purpose resources for pre/post-processing. The tight coupling with system memory and CPU enables better integration than discrete accelerators.

The Hailo-10H takes a different approach: it's a dedicated AI accelerator chip with its own memory interface (LPDDR4/4X), delivering 40 TOPS at INT4 precision while consuming only 2.5W typical power. The Hailo-10H is specifically designed for generative AI workloads including LLMs, VLMs, and Stable Diffusion, with a direct DDR interface that allows it to scale for large models. This dedicated architecture with additional memory represents another evolution in edge AI acceleration.

Both approaches offer advantages: NPUs provide tight system integration, while dedicated accelerators like Hailo-10H offer raw performance and memory bandwidth. They might represent the future of edge AI: specialized enough to be efficient, but flexible enough to handle diverse workloads.

## Conclusion

Our benchmark demonstrates that hardware selection matters more than raw computational power. For small Transformer models, CPUs with large caches and specialized edge devices can outperform high-end GPUs due to lower latency (cache vs PCIe), reduced overhead (no kernel launch costs), and better power efficiency.

The results challenge the assumption that GPUs are always faster. Instead, they suggest that workload characteristics should drive hardware selection, not ecosystem momentum.

As specialized hardware matures and software abstraction improves, we'll see more diverse hardware choices in ML deployment. CUDA's dominance is real, but it's not permanent. The future belongs to hardware that matches workload requirements, not just hardware that's widely available.

The benchmark methodology — using Docker and NVIDIA Container Toolkit for reproducibility, measuring resource consumption inside containers, and ensuring identical models across platforms — provides a framework for making informed hardware decisions. When you control for variables and measure systematically, the results can challenge conventional wisdom.

### Caveats and Limitations

It's important to acknowledge the limitations of this benchmark. The results are specific to small models (10M parameters) that fit entirely in CPU cache. For larger models that exceed cache capacity, GPUs would likely regain their advantage. The benchmark uses 8-bit quantization, which is necessary for edge devices but may not represent all use cases. The workload is inference-only — training workloads would show different characteristics entirely.

The GPU's poor performance here is partly due to TensorFlow overhead and the small workload size. With larger batch sizes or larger models, GPU parallelism would become more advantageous. The 8% GPU utilization indicates the workload is too small to fully utilize the hardware.

These caveats don't invalidate the results — they contextualize them. For edge deployment of small, quantized models, CPUs and specialized accelerators can indeed outperform GPUs. But the hardware landscape is nuanced, and workload characteristics matter enormously.

The takeaway is clear: don't assume GPUs are always faster. Measure, compare, and choose hardware based on workload characteristics, not ecosystem momentum.

**Sources:**

- Benchmark Code: https://github.com/[your-repo]/LLMs-FPGA
- Results and Analysis: See the Jupyter notebook in the `analysis/` directory
- Google Coral Edge TPU: https://coral.ai/
- Hailo-10H AI Accelerator: https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

