#!/usr/bin/env python3
"""
Download the face swap model from alternative sources.
"""

import os
import urllib.request
import hashlib
from pathlib import Path

def download_with_progress(url, destination):
    """Download file with progress bar."""
    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        print(f"Downloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
    
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, destination, reporthook=download_hook)
    print("\nDownload complete!")

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "inswapper_128.onnx"
    
    # Alternative download URLs
    urls = [
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
    ]
    
    for url in urls:
        try:
            print(f"\nTrying to download from: {url}")
            download_with_progress(url, str(model_path))
            
            # Check file size (should be around 500MB)
            file_size = model_path.stat().st_size
            print(f"File size: {file_size / 1024 / 1024:.1f} MB")
            
            if file_size > 100 * 1024 * 1024:  # At least 100MB
                print("Model downloaded successfully!")
                return True
            else:
                print("Downloaded file is too small, trying next URL...")
                model_path.unlink()
                
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if model_path.exists():
                model_path.unlink()
    
    print("\n❌ Failed to download the model automatically.")
    print("\n📥 Please download manually:")
    print("1. Go to: https://huggingface.co/deepinsight/inswapper/tree/main")
    print("2. Download 'inswapper_128.onnx' (about 500MB)")
    print(f"3. Place it in: {model_path.absolute()}")
    
    return False

if __name__ == "__main__":
    main() 