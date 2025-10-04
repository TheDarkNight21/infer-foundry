"""
CLI module for InferFoundry
"""

import click
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import onnxruntime as ort
import psutil
import torch
from onnx2torch import convert


class ONNXBenchmarker:
    """ONNX model benchmarking class using onnx2torch conversion"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None  # Keep for metadata extraction
        self.torch_model = None  # PyTorch model after conversion
        self.input_shape = None
        self.input_name = None
        self.output_name = None
        self.model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def check_external_data_files(self) -> bool:
        """Check if external data files exist"""
        data_file = f"{self.model_path}.data"
        return os.path.exists(data_file)
        
    def load_model(self) -> None:
        """Load ONNX model, extract metadata, and convert to PyTorch"""
        try:
            # Check for external data files first
            if not self.check_external_data_files():
                print(f"⚠️  Checking for external data files...")
                print(f"   Looking for: {self.model_path}.data")
                print(f"   File not found - model may use embedded weights or external data is missing")
            
            # Try to load the ONNX model for metadata extraction
            try:
                self.session = ort.InferenceSession(self.model_path)
                print(f"✓ ONNX Runtime loaded model successfully")
            except Exception as onnx_error:
                # Check if it's an external data issue
                if ".onnx.data" in str(onnx_error) or "external data" in str(onnx_error).lower():
                    print(f"❌ ONNX model has external data dependencies but .onnx.data file is missing")
                    print(f"   Error: {onnx_error}")
                    print(f"")
                    print(f"🔧 SOLUTION:")
                    print(f"   This ONNX model was saved with external data storage.")
                    print(f"   You need both files:")
                    print(f"   1. {self.model_path}")
                    print(f"   2. {self.model_path}.data")
                    print(f"")
                    print(f"   Please download the .onnx.data file and place it in the same directory.")
                    print(f"   The .onnx.data file contains the model weights and is required for loading.")
                    print(f"")
                    print(f"   Alternative: If you have access to the original model, try saving it")
                    print(f"   with embedded weights instead of external data storage.")
                    raise RuntimeError(f"Missing .onnx.data file. Please download and place {self.model_path}.data in the same directory.")
                else:
                    raise onnx_error
            
            # Get model metadata
            self.model_name = Path(self.model_path).stem
            
            # Get input/output information
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            
            # Get output name
            output_info = self.session.get_outputs()[0]
            self.output_name = output_info.name
            
            print(f"✓ Model loaded: {self.model_name}")
            print(f"✓ Input shape: {self.input_shape}")
            print(f"✓ Input name: {self.input_name}")
            print(f"✓ Output name: {self.output_name}")
            
            # Convert ONNX model to PyTorch
            print(f"🔄 Converting ONNX model to PyTorch...")
            try:
                # Try to convert with onnx2torch
                self.torch_model = convert(self.model_path)
                self.torch_model.to(self.device)
                self.torch_model.eval()
                print(f"✓ Model converted to PyTorch and moved to {self.device}")
            except Exception as convert_error:
                # Check for specific error types
                error_str = str(convert_error).lower()
                
                if "permission denied" in error_str or "errno 13" in error_str:
                    print(f"⚠️  Permission denied during PyTorch conversion")
                    print(f"   Error: {convert_error}")
                    print(f"   This may be due to temporary file creation restrictions")
                    print(f"   Trying with different temporary directory...")
                    
                    # Try with a different temporary directory
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Set temporary directory environment variable
                            old_temp = os.environ.get('TMPDIR', os.environ.get('TEMP', os.environ.get('TMP')))
                            os.environ['TMPDIR'] = temp_dir
                            os.environ['TEMP'] = temp_dir
                            os.environ['TMP'] = temp_dir
                            
                            self.torch_model = convert(self.model_path)
                            self.torch_model.to(self.device)
                            self.torch_model.eval()
                            print(f"✓ Model converted to PyTorch using alternative temp directory")
                            
                            # Restore original temp directory
                            if old_temp:
                                os.environ['TMPDIR'] = old_temp
                                os.environ['TEMP'] = old_temp
                                os.environ['TMP'] = old_temp
                    except Exception as temp_error:
                        print(f"   Alternative temp directory also failed: {temp_error}")
                        print(f"   Falling back to ONNX Runtime for inference...")
                        print(f"   Note: ONNX Runtime will be used instead of PyTorch")
                        self.torch_model = None
                elif "external data" in error_str or ".onnx.data" in error_str:
                    print(f"⚠️  ONNX model may have external data dependencies")
                    print(f"   Error: {convert_error}")
                    print(f"   Trying alternative approach...")
                    print(f"   Falling back to ONNX Runtime for inference...")
                    print(f"   Note: This may be due to missing .onnx.data file or external data dependencies")
                    self.torch_model = None
                else:
                    print(f"⚠️  PyTorch conversion failed: {convert_error}")
                    print(f"   Falling back to ONNX Runtime for inference...")
                    print(f"   Note: ONNX Runtime will be used instead of PyTorch")
                    self.torch_model = None
            
        except Exception as e:
            raise RuntimeError(f"Failed to load and convert model: {e}")
    
    def prepare_dummy_input(self) -> np.ndarray:
        """Prepare dummy input tensor matching model's input shape"""
        # Handle dynamic shapes by using reasonable defaults
        shape = []
        for dim in self.input_shape:
            if isinstance(dim, str) or dim == -1:
                # Use default values for dynamic dimensions
                if len(shape) == 0:  # batch size
                    shape.append(1)
                elif len(shape) == 1:  # sequence length
                    shape.append(512)  # reasonable default for text
                else:  # hidden dimensions
                    shape.append(768)  # reasonable default
            else:
                shape.append(dim)
        
        # Check if this is a language model (GPT-2, etc.) that expects integer tokens
        is_language_model = (
            'gpt' in self.model_name.lower() or 
            'input1' in self.input_name.lower() or
            'input_ids' in self.input_name.lower() or
            'token' in self.input_name.lower()
        )
        
        if is_language_model:
            # For language models, create integer token IDs
            # Use a reasonable vocabulary size (GPT-2 has ~50k tokens)
            vocab_size = 50257  # GPT-2 vocabulary size
            dummy_input = np.random.randint(0, vocab_size, size=shape, dtype=np.int64)
            print(f"   Using integer tokens for language model (vocab_size: {vocab_size})")
        else:
            # For other models, use float32
            dummy_input = np.random.randn(*shape).astype(np.float32)
            print(f"   Using float32 input for model")
        
        return dummy_input
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage if available"""
        gpu_info = {}
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            gpu_info['cuda_available'] = False
            
        return gpu_info
    
    def run_inference(self, input_data: np.ndarray) -> Tuple[float, Any]:
        """Run single inference using PyTorch model or ONNX Runtime fallback"""
        start_time = time.perf_counter()
        
        if self.torch_model is not None:
            # Use PyTorch model
            input_tensor = torch.from_numpy(input_data).to(self.device)
            
            with torch.no_grad():
                output = self.torch_model(input_tensor)
            
            # Convert output back to numpy for consistency
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            elif isinstance(output, (list, tuple)):
                output = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in output]
        else:
            # Fallback to ONNX Runtime
            # Ensure input data type matches what the model expects
            if input_data.dtype == np.int64:
                # For language models with integer inputs
                output = self.session.run([self.output_name], {self.input_name: input_data})
            else:
                # For other models with float inputs
                output = self.session.run([self.output_name], {self.input_name: input_data})
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return latency_ms, output
    
    def benchmark(self, warmup_runs: int = 3, timed_runs: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark"""
        print(f"\n🚀 Starting benchmark...")
        print(f"   Warmup runs: {warmup_runs}")
        print(f"   Timed runs: {timed_runs}")
        
        # Prepare dummy input
        dummy_input = self.prepare_dummy_input()
        
        # Warmup runs
        print(f"\n🔥 Warming up...")
        for i in range(warmup_runs):
            self.run_inference(dummy_input)
            if (i + 1) % 10 == 0:
                print(f"   Warmup {i + 1}/{warmup_runs}")
        
        # Timed runs
        print(f"\n⏱️  Running timed inference...")
        latencies = []
        
        for i in range(timed_runs):
            latency, _ = self.run_inference(dummy_input)
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"   Completed {i + 1}/{timed_runs} runs")
        
        # Calculate metrics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Calculate throughput (assuming 1 token per inference for simplicity)
        # In a real scenario, you'd need to know the actual token count
        throughput = 1000 / avg_latency  # tokens per second
        
        # Get GPU memory info
        gpu_info = self.get_gpu_memory_usage()
        
        # Get system memory info
        memory_info = psutil.virtual_memory()
        
        # Determine runtime based on whether conversion was successful
        if self.torch_model is not None:
            runtime = 'PyTorch (converted from ONNX)'
        else:
            runtime = 'ONNX Runtime (PyTorch conversion failed)'
        
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'runtime': runtime,
            'device': str(self.device) if self.torch_model is not None else 'CPU (ONNX Runtime)',
            'input_shape': self.input_shape,
            'warmup_runs': warmup_runs,
            'timed_runs': timed_runs,
            'latency': {
                'average_ms': round(avg_latency, 2),
                'std_ms': round(std_latency, 2),
                'min_ms': round(min_latency, 2),
                'max_ms': round(max_latency, 2)
            },
            'throughput': {
                'tokens_per_second': round(throughput, 2)
            },
            'gpu_memory': gpu_info,
            'system_memory': {
                'total_gb': round(memory_info.total / 1024**3, 2),
                'available_gb': round(memory_info.available / 1024**3, 2),
                'used_percent': memory_info.percent
            }
        }
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark report"""
        print(f"\n{'='*50}")
        print(f"📊 BENCHMARK REPORT")
        print(f"{'='*50}")
        print(f"Model: {results['model_name']}")
        print(f"Runtime: {results['runtime']}")
        print(f"Device: {results['device']}")
        print(f"Input Shape: {results['input_shape']}")
        print(f"")
        print(f"⏱️  LATENCY:")
        print(f"   Average: {results['latency']['average_ms']}ms")
        print(f"   Std Dev: {results['latency']['std_ms']}ms")
        print(f"   Min: {results['latency']['min_ms']}ms")
        print(f"   Max: {results['latency']['max_ms']}ms")
        print(f"")
        print(f"🚀 THROUGHPUT:")
        print(f"   {results['throughput']['tokens_per_second']} tokens/sec")
        print(f"")
        
        if results['gpu_memory']['cuda_available']:
            print(f"🎮 GPU MEMORY:")
            print(f"   Allocated: {results['gpu_memory']['gpu_memory_allocated']:.2f} GB")
            print(f"   Reserved: {results['gpu_memory']['gpu_memory_reserved']:.2f} GB")
        else:
            print(f"🎮 GPU: Not available")
        
        print(f"💾 SYSTEM MEMORY:")
        print(f"   Used: {results['system_memory']['used_percent']:.1f}%")
        print(f"   Available: {results['system_memory']['available_gb']:.2f} GB")
        print(f"{'='*50}")
    
    def save_json_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save results to JSON file"""
        if output_path is None:
            output_path = f"{self.model_name}_benchmark.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Report saved to: {output_path}")
        return output_path


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """InferFoundry - ONNX Model Benchmarking Tool (converts ONNX to PyTorch)"""
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model file (will be converted to PyTorch)')
@click.option('--warmup', '-w', default=3, help='Number of warmup runs')
@click.option('--runs', '-r', default=100, help='Number of timed runs')
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def benchmark(model, warmup, runs, output, quiet):
    """Benchmark an ONNX model (converted to PyTorch for execution)"""
    
    # Validate model file exists
    if not os.path.exists(model):
        click.echo(f"❌ Error: Model file '{model}' not found", err=True)
        return
    
    try:
        # Create benchmarker
        benchmarker = ONNXBenchmarker(model)
        
        # Load model
        if not quiet:
            click.echo("🔄 Loading model...")
        benchmarker.load_model()
        
        # Run benchmark
        results = benchmarker.benchmark(warmup_runs=warmup, timed_runs=runs)
        
        # Print report
        if not quiet:
            benchmarker.print_report(results)
        else:
            # Quiet mode - just show key metrics
            click.echo(f"Model: {results['model_name']}")
            click.echo(f"Runtime: {results['runtime']}")
            click.echo(f"Avg Latency: {results['latency']['average_ms']}ms")
            click.echo(f"Throughput: {results['throughput']['tokens_per_second']} tokens/sec")
        
        # Save JSON report
        if output:
            benchmarker.save_json_report(results, output)
        elif not quiet:
            benchmarker.save_json_report(results)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return


if __name__ == '__main__':
    cli()
