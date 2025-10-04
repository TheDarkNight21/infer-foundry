#!/usr/bin/env python3
"""
Simple download script for tiny ONNX models with embedded weights
"""

import os
import onnxruntime as ort
from pathlib import Path
from onnxruntime.datasets import get_example
import shutil

def main():
    """Download small ONNX models for testing"""
    print("🚀 Downloading tiny ONNX models with embedded weights...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download sigmoid model (very small, embedded weights)
    print("📥 Downloading sigmoid model...")
    try:
        model_path = get_example("sigmoid.onnx")
        dest_path = models_dir / "sigmoid.onnx"
        shutil.copy2(model_path, dest_path)
        print(f"✓ Sigmoid model saved to: {dest_path}")
        
        # Verify it works
        session = ort.InferenceSession(str(dest_path))
        print(f"✓ Model verified - Input: {session.get_inputs()[0].shape}")
        
    except Exception as e:
        print(f"❌ Failed to download sigmoid model: {e}")
    
    print("\n🎉 Ready to test!")
    print("Run: inferfoundry benchmark --model ./models/sigmoid.onnx")

if __name__ == "__main__":
    main()
