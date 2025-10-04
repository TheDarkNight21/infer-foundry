# ONNX Model Download Scripts

This directory contains scripts to download small ONNX models with embedded weights for testing the InferFoundry CLI.

## Quick Start

### Simple Download (Recommended)
```bash
python simple_download.py
```

This will download a tiny sigmoid model that's perfect for testing.

### Full Download
```bash
python download_models.py
```

This will download multiple small models from the ONNX Model Zoo.

## Available Models

### Simple Download Script
- **gpt2-small.onnx** - Small GPT-2 language model with embedded weights
  - Better for LLM testing than sigmoid
  - No external data dependencies
  - More realistic language model for benchmarking
  - Fallback to sigmoid.onnx if GPT-2 download fails

### Full Download Script
- **gpt2-small.onnx** - Small GPT-2 language model (primary recommendation)
- **sigmoid.onnx** - Simple sigmoid function model
- **linear_regression.onnx** - Linear regression model
- **mnist.onnx** - MNIST digit classification model
- **tiny_yolov3.onnx** - Tiny YOLOv3 object detection model

## Usage

After downloading models, you can test them with:

```bash
# Test with GPT-2 small model (recommended for LLM testing)
inferfoundry benchmark --model ./models/gpt2-small.onnx

# Test with other models
inferfoundry benchmark --model ./models/sigmoid.onnx
inferfoundry benchmark --model ./models/mnist.onnx
inferfoundry benchmark --model ./models/tiny_yolov3.onnx
```

## Why These Models?

These models are chosen because they:
- ✅ Have embedded weights (no .onnx.data files needed)
- ✅ Are small and fast to download
- ✅ Work reliably with onnx2torch conversion
- ✅ Are perfect for testing and benchmarking

## Troubleshooting

If you encounter issues:
1. Make sure you have internet connectivity
2. Check that the models directory was created
3. Verify the model files exist before running benchmarks
4. Use the simple download script first for basic testing
