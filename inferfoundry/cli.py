"""
CLI module for InferFoundry
"""

import click
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import onnxruntime as ort
import psutil
import torch


class ONNXBenchmarker:
    """ONNX model benchmarking class"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_shape = None
        self.input_name = None
        self.output_name = None
        self.model_name = None
        
    def load_model(self) -> None:
        """Load ONNX model and extract metadata"""
        try:
            # Load the ONNX model
            self.session = ort.InferenceSession(self.model_path)
            
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
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
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
        
        # Create dummy input
        dummy_input = np.random.randn(*shape).astype(np.float32)
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
        """Run single inference and return timing + output"""
        start_time = time.perf_counter()
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
        
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'runtime': 'ONNX',
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
    """InferFoundry - ONNX Model Benchmarking Tool"""
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model file')
@click.option('--warmup', '-w', default=3, help='Number of warmup runs')
@click.option('--runs', '-r', default=100, help='Number of timed runs')
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def benchmark(model, warmup, runs, output, quiet):
    """Benchmark an ONNX model"""
    
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
