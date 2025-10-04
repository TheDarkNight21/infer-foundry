# InferFoundry

A powerful CLI tool for benchmarking ONNX models with comprehensive performance metrics across multiple runtimes. Supports ONNX Runtime and TensorRT for performance comparison.

## Features

- 🚀 **Multi-Runtime Support**: Compare ONNX Runtime vs TensorRT performance
- ⚡ **Fast Benchmarking**: Quick and accurate performance measurement
- 🔄 **ONNX to PyTorch**: Automatically converts ONNX models to PyTorch for execution
- 🎯 **TensorRT Integration**: Direct TensorRT engine building and execution
- 📊 **Comprehensive Metrics**: Latency, throughput, and memory usage
- 🎮 **GPU Support**: CUDA acceleration for both PyTorch, ONNX Runtime, and TensorRT
- 📄 **Multiple Outputs**: Console reports and JSON export
- ⚡ **Easy to Use**: Simple CLI interface with sensible defaults
- 🛡️ **Robust Error Handling**: Clear error messages for missing external data files
- 📈 **Performance Comparison**: Side-by-side runtime comparison with speedup metrics

## Installation

### GPU Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd infer-foundry

# Install CUDA-enabled dependencies
pip install -r requirements.txt

# Or install manually for better control
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
pip install tensorrt pycuda
pip install -r requirements.txt
```

**Note**: TensorRT requires NVIDIA GPU and specific CUDA versions. If TensorRT is not available, the tool will fall back to ONNX Runtime only.

### CPU Installation (Fallback)

```bash
# Clone the repository
git clone <repository-url>
cd infer-foundry

# Install CPU-only dependencies
pip install -r requirements-cpu.txt
```

### Development Installation

```bash
# Install in development mode
pip install -e .
```

## Quick Start with Test Models

To get started quickly, download some small test models:

```bash
# Download tiny ONNX models with embedded weights
python simple_download.py

# Or download multiple models
python download_models.py
```

Then test the CLI:
```bash
inferfoundry benchmark --model ./models/gpt2-small.onnx
```

See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for more details about available test models.

## Usage

### Basic Benchmarking

```bash
# Benchmark an ONNX model with ONNX Runtime
inferfoundry benchmark --model ./tinyllama.onnx

# Compare ONNX Runtime vs TensorRT
inferfoundry benchmark --model ./tinyllama.onnx --runtimes onnx,tensorrt
```

### Advanced Options

```bash
# Custom warmup and run counts
inferfoundry benchmark --model ./model.onnx --warmup 5 --runs 200

# Test only TensorRT runtime
inferfoundry benchmark --model ./model.onnx --runtimes tensorrt

# Save results to specific file
inferfoundry benchmark --model ./model.onnx --runtimes onnx,tensorrt --output results.json

# Quiet mode (minimal output)
inferfoundry benchmark --model ./model.onnx --runtimes onnx,tensorrt --quiet
```

### Report Command

```bash
# Display a report from saved JSON results
inferfoundry report --path results.json
```

### Command Line Options

- `--model, -m`: Path to ONNX model file (required)
- `--runtimes, -r`: Comma-separated list of runtimes to test (e.g., onnx,tensorrt) (default: onnx)
- `--warmup, -w`: Number of warmup runs (default: 3)
- `--runs, -n`: Number of timed runs (default: 100)
- `--output, -o`: Output JSON file path (optional)
- `--quiet, -q`: Suppress progress output (flag)

### Report Command Options

- `--path, -p`: Path to JSON results file (required)

## Example Output

### Single Runtime (ONNX)
```
🔄 Loading model...
✓ Model loaded: tinyllama
✓ Input shape: [1, 512]
✓ Input name: input_ids
✓ Output name: logits

🚀 Starting benchmark...
   Warmup runs: 3
   Timed runs: 100

🔥 Warming up...
   Warmup 3/3

⏱️  Running timed inference...
   Completed 20/100 runs
   Completed 40/100 runs
   Completed 60/100 runs
   Completed 80/100 runs
   Completed 100/100 runs

==================================================
📊 BENCHMARK REPORT
==================================================
Model: tinyllama
Runtime: ONNX
Input Shape: [1, 512]

⏱️  LATENCY:
   Average: 13.2ms
   Std Dev: 1.8ms
   Min: 10.1ms
   Max: 18.5ms

🚀 THROUGHPUT:
   75 tokens/sec

🎮 GPU MEMORY:
   Allocated: 2.1 GB
   Reserved: 2.5 GB

💾 SYSTEM MEMORY:
   Used: 45.2%
   Available: 8.7 GB
==================================================
📄 Report saved to: tinyllama_benchmark.json
```

### Multi-Runtime Comparison (ONNX vs TensorRT)
```
🔄 Running comparison across runtimes: onnx, tensorrt

==================================================
🚀 Testing ONNX runtime
==================================================
✓ ONNX benchmark completed

==================================================
🚀 Testing TENSORRT runtime
==================================================
🔄 Building TensorRT engine from ONNX model...
✓ TensorRT engine built and saved to: tinyllama.trt
✓ TensorRT benchmark completed

============================================================
📊 RUNTIME COMPARISON REPORT
============================================================
Model: tinyllama.onnx

Runtime              Latency (ms)     Speedup    
---------------------------------------------
TensorRT             8.1              1.63x      
ONNX                 13.2             baseline   
============================================================
📄 Comparison report saved to: tinyllama_comparison.json
```

## JSON Output

The tool generates detailed JSON reports with all metrics:

### Single Runtime Results
```json
{
  "model_name": "tinyllama",
  "model_path": "./tinyllama.onnx",
  "runtime": "ONNX",
  "input_shape": [1, 512],
  "warmup_runs": 3,
  "timed_runs": 100,
  "latency": {
    "average_ms": 13.2,
    "std_ms": 1.8,
    "min_ms": 10.1,
    "max_ms": 18.5
  },
  "throughput": {
    "tokens_per_second": 75.0
  },
  "gpu_memory": {
    "cuda_available": true,
    "gpu_memory_allocated": 2.1,
    "gpu_memory_reserved": 2.5
  },
  "system_memory": {
    "total_gb": 16.0,
    "available_gb": 8.7,
    "used_percent": 45.2
  }
}
```

### Multi-Runtime Comparison Results
```json
{
  "model": "tinyllama.onnx",
  "results": [
    {"runtime": "onnx", "latency_ms": 13.2},
    {"runtime": "tensorrt", "latency_ms": 8.1}
  ]
}
```

## Requirements

- Python 3.8+
- ONNX Runtime
- NumPy
- PyTorch (for GPU memory monitoring)
- Click (for CLI)
- psutil (for system monitoring)

## License

MIT License - see LICENSE file for details.
