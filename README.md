# InferFoundry

A powerful CLI tool for benchmarking ONNX models with comprehensive performance metrics.

## Features

- 🚀 **Fast Benchmarking**: Quick and accurate performance measurement
- 📊 **Comprehensive Metrics**: Latency, throughput, and memory usage
- 🎮 **GPU Support**: CUDA memory monitoring when available
- 📄 **Multiple Outputs**: Console reports and JSON export
- ⚡ **Easy to Use**: Simple CLI interface with sensible defaults

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd infer-foundry

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Usage

### Basic Benchmarking

```bash
# Benchmark an ONNX model
inferfoundry benchmark --model ./tinyllama.onnx
```

### Advanced Options

```bash
# Custom warmup and run counts
inferfoundry benchmark --model ./model.onnx --warmup 5 --runs 200

# Save results to specific file
inferfoundry benchmark --model ./model.onnx --output results.json

# Quiet mode (minimal output)
inferfoundry benchmark --model ./model.onnx --quiet
```

### Command Line Options

- `--model, -m`: Path to ONNX model file (required)
- `--warmup, -w`: Number of warmup runs (default: 3)
- `--runs, -r`: Number of timed runs (default: 100)
- `--output, -o`: Output JSON file path (optional)
- `--quiet, -q`: Suppress progress output (flag)

## Example Output

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

## JSON Output

The tool also generates detailed JSON reports with all metrics:

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

## Requirements

- Python 3.8+
- ONNX Runtime
- NumPy
- PyTorch (for GPU memory monitoring)
- Click (for CLI)
- psutil (for system monitoring)

## License

MIT License - see LICENSE file for details.
