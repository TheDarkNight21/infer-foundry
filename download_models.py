#!/usr/bin/env python3
"""
Download script for small ONNX models with embedded weights
These models are perfect for testing the InferFoundry CLI
"""

import os
import requests
import onnxruntime as ort
from pathlib import Path
from onnxruntime.datasets import get_example
import tempfile
import shutil

def create_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def download_gpt2_small_model(models_dir):
    """Download a small GPT-2 model from ONNX Model Zoo"""
    print("📥 Downloading GPT-2 model...")
    print("   Note: This model is ~523MB (larger than sigmoid but still manageable)")
    try:
        # Small GPT-2 model from ONNX Model Zoo with embedded weights
        model_url = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.onnx"
        dest_path = models_dir / "gpt2-small.onnx"
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ GPT-2 small model saved to: {dest_path}")
        print(f"✓ Model type: Small GPT-2 language model (better for LLM testing)")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download GPT-2 small model: {e}")
        return None

def download_sigmoid_model(models_dir):
    """Download the sigmoid model from ONNX Runtime examples"""
    print("📥 Downloading sigmoid model...")
    try:
        # Get the sigmoid model from onnxruntime examples
        model_path = get_example("sigmoid.onnx")
        
        # Copy to our models directory
        dest_path = models_dir / "sigmoid.onnx"
        shutil.copy2(model_path, dest_path)
        
        print(f"✓ Sigmoid model saved to: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download sigmoid model: {e}")
        return None

def download_linear_regression_model(models_dir):
    """Download a simple linear regression model"""
    print("📥 Downloading linear regression model...")
    try:
        # Get the linear regression model from onnxruntime examples
        model_path = get_example("linear_regression.onnx")
        
        # Copy to our models directory
        dest_path = models_dir / "linear_regression.onnx"
        shutil.copy2(model_path, dest_path)
        
        print(f"✓ Linear regression model saved to: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download linear regression model: {e}")
        return None

def download_tiny_yolov3_model(models_dir):
    """Download a tiny YOLOv3 model from ONNX Model Zoo"""
    print("📥 Downloading Tiny YOLOv3 model...")
    try:
        # Tiny YOLOv3 model URL from ONNX Model Zoo
        url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        dest_path = models_dir / "tiny_yolov3.onnx"
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Tiny YOLOv3 model saved to: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download Tiny YOLOv3 model: {e}")
        return None

def download_mnist_model(models_dir):
    """Download a simple MNIST model"""
    print("📥 Downloading MNIST model...")
    try:
        # Simple MNIST model URL
        url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        dest_path = models_dir / "mnist.onnx"
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ MNIST model saved to: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"❌ Failed to download MNIST model: {e}")
        return None

def verify_model(model_path):
    """Verify that a model can be loaded successfully"""
    try:
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"   ✓ Model verified - Input: {input_info.shape}, Output: {output_info.shape}")
        return True
    except Exception as e:
        print(f"   ❌ Model verification failed: {e}")
        return False

def main():
    """Main download function"""
    print("🚀 Downloading small ONNX models with embedded weights...")
    print("   These models are perfect for testing InferFoundry CLI")
    print()
    
    # Create models directory
    models_dir = create_models_directory()
    print(f"📁 Models directory: {models_dir.absolute()}")
    print()
    
    # List of download functions
    download_functions = [
        download_gpt2_small_model,  # Small language model (better for LLM testing)
        download_sigmoid_model,
        download_linear_regression_model,
        download_mnist_model,
        download_tiny_yolov3_model,
    ]
    
    successful_downloads = []
    
    # Download models
    for download_func in download_functions:
        try:
            model_path = download_func(models_dir)
            if model_path and os.path.exists(model_path):
                successful_downloads.append(model_path)
            print()
        except Exception as e:
            print(f"❌ Error in {download_func.__name__}: {e}")
            print()
    
    # Verify downloaded models
    print("🔍 Verifying downloaded models...")
    verified_models = []
    
    for model_path in successful_downloads:
        print(f"Verifying {Path(model_path).name}...")
        if verify_model(model_path):
            verified_models.append(model_path)
        print()
    
    # Summary
    print("="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total models downloaded: {len(successful_downloads)}")
    print(f"Verified models: {len(verified_models)}")
    print()
    
    if verified_models:
        print("✅ Ready to use models:")
        for model_path in verified_models:
            model_name = Path(model_path).stem
            print(f"   • {model_path}")
            print(f"     Command: inferfoundry benchmark --model {model_path}")
        print()
        print("🎉 You can now test the CLI with any of these models!")
    else:
        print("❌ No models were successfully downloaded and verified.")
        print("   Please check your internet connection and try again.")
    
    print("="*60)

if __name__ == "__main__":
    main()
