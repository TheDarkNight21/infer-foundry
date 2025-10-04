#!/usr/bin/env python3
"""
Simple download script for tiny ONNX models with embedded weights
"""

import os
import requests
import onnxruntime as ort
from pathlib import Path

def main():
    """Download small ONNX models for testing"""
    print("🚀 Downloading tiny ONNX models with embedded weights...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download a small GPT-2 model (better than sigmoid for LLM testing)
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
        
        # Verify it works
        session = ort.InferenceSession(str(dest_path))
        input_info = session.get_inputs()[0]
        print(f"✓ Model verified - Input: {input_info.name} with shape {input_info.shape}")
        print(f"✓ Model type: Small GPT-2 language model (better for LLM testing)")
        
    except Exception as e:
        print(f"❌ Failed to download GPT-2 model: {e}")
        print("   Falling back to sigmoid model...")
        
        # Fallback to sigmoid model
        try:
            from onnxruntime.datasets import get_example
            import shutil
            
            model_path = get_example("sigmoid.onnx")
            dest_path = models_dir / "sigmoid.onnx"
            shutil.copy2(model_path, dest_path)
            print(f"✓ Sigmoid model saved to: {dest_path}")
            
            # Verify it works
            session = ort.InferenceSession(str(dest_path))
            print(f"✓ Model verified - Input: {session.get_inputs()[0].shape}")
            
        except Exception as e2:
            print(f"❌ Failed to download sigmoid model: {e2}")
    
    print("\n🎉 Ready to test!")
    print("Run: inferfoundry benchmark --model ./models/gpt2-small.onnx")
    print("   (or ./models/sigmoid.onnx if GPT-2 download failed)")

if __name__ == "__main__":
    main()
