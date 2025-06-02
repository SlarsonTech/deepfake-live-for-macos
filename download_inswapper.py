#!/usr/bin/env python3
"""
Download inswapper_128.onnx model from Google Drive.
"""

import os
import requests
from pathlib import Path

def download_from_google_drive(file_id, destination):
    """Download file from Google Drive using direct link."""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Extract confirmation token from Google Drive response."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save response content to file with progress."""
    CHUNK_SIZE = 32768
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    print()  # New line after download

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "inswapper_128.onnx"
    
    # Google Drive file ID from the GitHub issue
    file_id = "1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF"
    
    print("Downloading inswapper_128.onnx from Google Drive...")
    print("This might take a few minutes as the file is about 500MB...")
    
    try:
        download_from_google_drive(file_id, str(model_path))
        
        # Check file size
        file_size = model_path.stat().st_size
        print(f"\nFile size: {file_size / 1024 / 1024:.1f} MB")
        
        if file_size > 100 * 1024 * 1024:  # At least 100MB
            print("✅ Model downloaded successfully!")
            print(f"Location: {model_path.absolute()}")
            
            # Calculate checksum
            import hashlib
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            print(f"SHA256: {sha256_hash.hexdigest()}")
            
            return True
        else:
            print("❌ Downloaded file is too small, download may have failed.")
            model_path.unlink()
            return False
            
    except Exception as e:
        print(f"Error downloading: {e}")
        if model_path.exists():
            model_path.unlink()
        return False

if __name__ == "__main__":
    main() 