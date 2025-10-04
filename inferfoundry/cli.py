"""
CLI module for InferFoundry
"""

import click
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import onnxruntime as ort
import psutil
import torch
from onnx2torch import convert

# TensorRT imports (with fallback if not available)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None


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
                # Configure ONNX Runtime providers (CUDA first if available)
                providers = []
                available_providers = ort.get_available_providers()
                
                if torch.cuda.is_available() and 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print(f"✓ CUDA available - using GPU acceleration")
                else:
                    providers = ['CPUExecutionProvider']
                    if torch.cuda.is_available():
                        print(f"⚠️  PyTorch CUDA available but ONNX Runtime CUDA not installed")
                        print(f"   Install with: pip install onnxruntime-gpu")
                    else:
                        print(f"✓ Using CPU execution")
                
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                print(f"✓ ONNX Runtime loaded model successfully")
                
                # Show which providers are actually being used
                active_providers = self.session.get_providers()
                print(f"✓ Available providers: {active_providers}")
                if 'CUDAExecutionProvider' in active_providers:
                    print(f"✓ Using GPU acceleration (CUDA)")
                    # Show GPU memory info
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"✓ GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB VRAM)")
                else:
                    print(f"✓ Using CPU execution")
                    print(f"⚠️  CUDA not available - check CUDA installation")
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
        
        try:
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
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            print(f"   ⚠️  Inference error after {latency_ms:.1f}ms: {e}")
            raise e
    
    def benchmark(self, warmup_runs: int = 3, timed_runs: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark"""
        print(f"\n🚀 Starting benchmark...")
        print(f"   Warmup runs: {warmup_runs}")
        print(f"   Timed runs: {timed_runs}")
        
        # Prepare dummy input
        dummy_input = self.prepare_dummy_input()
        
        # Warmup runs
        print(f"\n🔥 Warming up...")
        print(f"   This may take a moment for large models...")
        for i in range(warmup_runs):
            start_warmup = time.perf_counter()
            self.run_inference(dummy_input)
            end_warmup = time.perf_counter()
            warmup_time = (end_warmup - start_warmup) * 1000
            
            if (i + 1) % 1 == 0:  # Show progress for every run
                print(f"   Warmup {i + 1}/{warmup_runs} - {warmup_time:.1f}ms")
        
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
            # Check if CUDA is being used for ONNX Runtime
            active_providers = self.session.get_providers()
            if 'CUDAExecutionProvider' in active_providers:
                runtime = 'ONNX Runtime with CUDA (PyTorch conversion failed)'
            else:
                runtime = 'ONNX Runtime CPU (PyTorch conversion failed)'
        
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'runtime': runtime,
            'device': str(self.device) if self.torch_model is not None else ('CUDA (ONNX Runtime)' if 'CUDAExecutionProvider' in self.session.get_providers() else 'CPU (ONNX Runtime)'),
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


class TensorRTRunner:
    """TensorRT model runner for ONNX models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.engine = None
        self.context = None
        self.input_shape = None
        self.input_name = None
        self.output_name = None
        self.model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Please install TensorRT and pycuda.")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TensorRT requires CUDA support.")
    
    def build_engine(self, onnx_path: str, engine_path: str = None) -> None:
        """Build TensorRT engine from ONNX model"""
        if engine_path is None:
            engine_path = onnx_path.replace('.onnx', '.trt')
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            print(f"✓ Loading existing TensorRT engine: {engine_path}")
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(engine_data)
        else:
            print(f"🔄 Building TensorRT engine from ONNX model...")
            print(f"   This may take several minutes for large models...")
            
            # Create TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("❌ Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Build engine
            self.engine = builder.build_engine(network, config)
            
            if self.engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(self.engine.serialize())
            
            print(f"✓ TensorRT engine built and saved to: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input/output information
        self.input_name = self.engine.get_binding_name(0)
        self.output_name = self.engine.get_binding_name(1)
        self.input_shape = self.engine.get_binding_shape(0)
        
        # Get model name
        self.model_name = Path(onnx_path).stem
        
        print(f"✓ TensorRT engine loaded: {self.model_name}")
        print(f"✓ Input shape: {self.input_shape}")
        print(f"✓ Input name: {self.input_name}")
        print(f"✓ Output name: {self.output_name}")
    
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
        
        # Check if this is a language model that expects integer tokens
        is_language_model = (
            'gpt' in self.model_name.lower() or 
            'input1' in self.input_name.lower() or
            'input_ids' in self.input_name.lower() or
            'token' in self.input_name.lower()
        )
        
        if is_language_model:
            # For language models, create integer token IDs
            vocab_size = 50257  # GPT-2 vocabulary size
            dummy_input = np.random.randint(0, vocab_size, size=shape, dtype=np.int32)
            print(f"   Using integer tokens for language model (vocab_size: {vocab_size})")
        else:
            # For other models, use float32
            dummy_input = np.random.randn(*shape).astype(np.float32)
            print(f"   Using float32 input for model")
        
        return dummy_input
    
    def run_inference(self, input_data: np.ndarray) -> Tuple[float, Any]:
        """Run single inference using TensorRT"""
        start_time = time.perf_counter()
        
        try:
            # Allocate GPU memory
            input_size = input_data.nbytes
            output_size = self.engine.get_binding_shape(1)[0] * 4  # Assume float32 output
            
            # Allocate host and device memory
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # Create CUDA stream
            stream = cuda.Stream()
            
            # Copy input data to GPU
            cuda.memcpy_htod_async(d_input, input_data, stream)
            
            # Run inference
            self.context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            
            # Copy output back to host
            h_output = np.empty(self.engine.get_binding_shape(1), dtype=np.float32)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            
            # Synchronize
            stream.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            return latency_ms, h_output
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            print(f"   ⚠️  TensorRT inference error after {latency_ms:.1f}ms: {e}")
            raise e
    
    def benchmark(self, warmup_runs: int = 3, timed_runs: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark"""
        print(f"\n🚀 Starting TensorRT benchmark...")
        print(f"   Warmup runs: {warmup_runs}")
        print(f"   Timed runs: {timed_runs}")
        
        # Prepare dummy input
        dummy_input = self.prepare_dummy_input()
        
        # Warmup runs
        print(f"\n🔥 Warming up TensorRT...")
        print(f"   This may take a moment for large models...")
        for i in range(warmup_runs):
            start_warmup = time.perf_counter()
            self.run_inference(dummy_input)
            end_warmup = time.perf_counter()
            warmup_time = (end_warmup - start_warmup) * 1000
            
            if (i + 1) % 1 == 0:  # Show progress for every run
                print(f"   Warmup {i + 1}/{warmup_runs} - {warmup_time:.1f}ms")
        
        # Timed runs
        print(f"\n⏱️  Running timed TensorRT inference...")
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
        
        # Calculate throughput
        throughput = 1000 / avg_latency  # tokens per second
        
        # Get GPU memory info
        gpu_info = self.get_gpu_memory_usage()
        
        # Get system memory info
        memory_info = psutil.virtual_memory()
        
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'runtime': 'TensorRT',
            'device': 'CUDA (TensorRT)',
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


class RuntimeComparator:
    """Compare performance between different runtimes"""
    
    def __init__(self, model_path: str, runtimes: List[str]):
        self.model_path = model_path
        self.runtimes = runtimes
        self.results = []
    
    def run_comparison(self, warmup_runs: int = 3, timed_runs: int = 100) -> Dict[str, Any]:
        """Run comparison across all specified runtimes"""
        print(f"\n🔄 Running comparison across runtimes: {', '.join(self.runtimes)}")
        
        for runtime in self.runtimes:
            print(f"\n{'='*50}")
            print(f"🚀 Testing {runtime.upper()} runtime")
            print(f"{'='*50}")
            
            try:
                if runtime.lower() == 'onnx':
                    benchmarker = ONNXBenchmarker(self.model_path)
                    benchmarker.load_model()
                    result = benchmarker.benchmark(warmup_runs=warmup_runs, timed_runs=timed_runs)
                elif runtime.lower() == 'tensorrt':
                    if not TENSORRT_AVAILABLE:
                        print(f"❌ TensorRT is not available. Skipping TensorRT runtime.")
                        continue
                    if not torch.cuda.is_available():
                        print(f"❌ CUDA is not available. Skipping TensorRT runtime.")
                        continue
                    runner = TensorRTRunner(self.model_path)
                    runner.build_engine(self.model_path)
                    result = runner.benchmark(warmup_runs=warmup_runs, timed_runs=timed_runs)
                else:
                    print(f"❌ Unknown runtime: {runtime}")
                    continue
                
                self.results.append(result)
                print(f"✓ {runtime.upper()} benchmark completed")
                
            except Exception as e:
                print(f"❌ {runtime.upper()} benchmark failed: {e}")
                continue
        
        if not self.results:
            raise RuntimeError("No runtimes completed successfully")
        
        # Create comparison summary
        comparison_result = {
            'model': Path(self.model_path).name,
            'results': []
        }
        
        for result in self.results:
            comparison_result['results'].append({
                'runtime': result['runtime'],
                'latency_ms': result['latency']['average_ms']
            })
        
        return comparison_result
    
    def print_comparison_report(self, comparison_result: Dict[str, Any]) -> None:
        """Print side-by-side comparison report"""
        print(f"\n{'='*60}")
        print(f"📊 RUNTIME COMPARISON REPORT")
        print(f"{'='*60}")
        print(f"Model: {comparison_result['model']}")
        print(f"")
        print(f"{'Runtime':<20} {'Latency (ms)':<15} {'Speedup':<10}")
        print(f"{'-'*45}")
        
        # Sort by latency for speedup calculation
        sorted_results = sorted(comparison_result['results'], key=lambda x: x['latency_ms'])
        baseline_latency = sorted_results[0]['latency_ms']
        
        for result in sorted_results:
            speedup = baseline_latency / result['latency_ms']
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
            print(f"{result['runtime']:<20} {result['latency_ms']:<15.2f} {speedup_str:<10}")
        
        print(f"{'='*60}")
    
    def save_comparison_json(self, comparison_result: Dict[str, Any], output_path: str = None) -> str:
        """Save comparison results to JSON file"""
        if output_path is None:
            output_path = f"{Path(self.model_path).stem}_comparison.json"
        
        with open(output_path, 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        print(f"📄 Comparison report saved to: {output_path}")
        return output_path


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """InferFoundry - Multi-Runtime ONNX Model Benchmarking Tool
    
    Supports ONNX Runtime and TensorRT for performance comparison.
    """
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model file')
@click.option('--runtimes', '-r', default='onnx', help='Comma-separated list of runtimes to test (e.g., onnx,tensorrt)')
@click.option('--warmup', '-w', default=3, help='Number of warmup runs')
@click.option('--runs', '-n', default=100, help='Number of timed runs')
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def benchmark(model, runtimes, warmup, runs, output, quiet):
    """Benchmark an ONNX model with multiple runtimes (ONNX, TensorRT)"""
    
    # Validate model file exists
    if not os.path.exists(model):
        click.echo(f"❌ Error: Model file '{model}' not found", err=True)
        return
    
    # Parse runtimes
    runtime_list = [r.strip().lower() for r in runtimes.split(',')]
    
    # Validate runtimes
    valid_runtimes = ['onnx', 'tensorrt']
    invalid_runtimes = [r for r in runtime_list if r not in valid_runtimes]
    if invalid_runtimes:
        click.echo(f"❌ Error: Invalid runtimes: {invalid_runtimes}. Valid options: {valid_runtimes}", err=True)
        return
    
    # Check TensorRT availability
    if 'tensorrt' in runtime_list and not TENSORRT_AVAILABLE:
        click.echo(f"⚠️  Warning: TensorRT is not available. Removing from runtime list.", err=True)
        runtime_list = [r for r in runtime_list if r != 'tensorrt']
    
    if not runtime_list:
        click.echo(f"❌ Error: No valid runtimes available", err=True)
        return
    
    try:
        if len(runtime_list) == 1:
            # Single runtime - use original behavior
            runtime = runtime_list[0]
            
            if runtime == 'onnx':
                benchmarker = ONNXBenchmarker(model)
                if not quiet:
                    click.echo("🔄 Loading ONNX model...")
                benchmarker.load_model()
                results = benchmarker.benchmark(warmup_runs=warmup, timed_runs=runs)
                
                if not quiet:
                    benchmarker.print_report(results)
                else:
                    click.echo(f"Model: {results['model_name']}")
                    click.echo(f"Runtime: {results['runtime']}")
                    click.echo(f"Avg Latency: {results['latency']['average_ms']}ms")
                    click.echo(f"Throughput: {results['throughput']['tokens_per_second']} tokens/sec")
                
                if output:
                    benchmarker.save_json_report(results, output)
                elif not quiet:
                    benchmarker.save_json_report(results)
            
            elif runtime == 'tensorrt':
                runner = TensorRTRunner(model)
                if not quiet:
                    click.echo("🔄 Building TensorRT engine...")
                runner.build_engine(model)
                results = runner.benchmark(warmup_runs=warmup, timed_runs=runs)
                
                if not quiet:
                    # Use similar report format as ONNX
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
                    print(f"{'='*50}")
                else:
                    click.echo(f"Model: {results['model_name']}")
                    click.echo(f"Runtime: {results['runtime']}")
                    click.echo(f"Avg Latency: {results['latency']['average_ms']}ms")
                    click.echo(f"Throughput: {results['throughput']['tokens_per_second']} tokens/sec")
                
                if output:
                    with open(output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📄 Report saved to: {output}")
                elif not quiet:
                    with open(f"{results['model_name']}_benchmark.json", 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📄 Report saved to: {results['model_name']}_benchmark.json")
        
        else:
            # Multiple runtimes - use comparison
            comparator = RuntimeComparator(model, runtime_list)
            comparison_result = comparator.run_comparison(warmup_runs=warmup, timed_runs=runs)
            
            if not quiet:
                comparator.print_comparison_report(comparison_result)
            else:
                # Quiet mode - just show key metrics
                for result in comparison_result['results']:
                    click.echo(f"{result['runtime']}: {result['latency_ms']}ms")
            
            if output:
                comparator.save_comparison_json(comparison_result, output)
            elif not quiet:
                comparator.save_comparison_json(comparison_result)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return


@cli.command()
@click.option('--path', '-p', required=True, help='Path to JSON results file')
def report(path):
    """Display a report from saved JSON results"""
    
    # Validate file exists
    if not os.path.exists(path):
        click.echo(f"❌ Error: Results file '{path}' not found", err=True)
        return
    
    try:
        # Load JSON results
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check if it's a comparison result or single result
        if 'results' in data and isinstance(data['results'], list):
            # Comparison result format
            print(f"\n{'='*60}")
            print(f"📊 RUNTIME COMPARISON REPORT")
            print(f"{'='*60}")
            print(f"Model: {data['model']}")
            print(f"")
            print(f"{'Runtime':<20} {'Latency (ms)':<15} {'Speedup':<10}")
            print(f"{'-'*45}")
            
            # Sort by latency for speedup calculation
            sorted_results = sorted(data['results'], key=lambda x: x['latency_ms'])
            baseline_latency = sorted_results[0]['latency_ms']
            
            for result in sorted_results:
                speedup = baseline_latency / result['latency_ms']
                speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
                print(f"{result['runtime']:<20} {result['latency_ms']:<15.2f} {speedup_str:<10}")
            
            print(f"{'='*60}")
        
        else:
            # Single result format
            print(f"\n{'='*50}")
            print(f"📊 BENCHMARK REPORT")
            print(f"{'='*50}")
            print(f"Model: {data['model_name']}")
            print(f"Runtime: {data['runtime']}")
            print(f"Device: {data['device']}")
            print(f"Input Shape: {data['input_shape']}")
            print(f"")
            print(f"⏱️  LATENCY:")
            print(f"   Average: {data['latency']['average_ms']}ms")
            print(f"   Std Dev: {data['latency']['std_ms']}ms")
            print(f"   Min: {data['latency']['min_ms']}ms")
            print(f"   Max: {data['latency']['max_ms']}ms")
            print(f"")
            print(f"🚀 THROUGHPUT:")
            print(f"   {data['throughput']['tokens_per_second']} tokens/sec")
            print(f"{'='*50}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        return


if __name__ == '__main__':
    cli()
