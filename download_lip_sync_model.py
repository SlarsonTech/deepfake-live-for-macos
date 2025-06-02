#!/usr/bin/env python3
"""
Download dlib shape predictor model for lip sync.
"""

import os
import urllib.request
import bz2
from pathlib import Path

def download_and_extract():
    """Download and extract the dlib shape predictor model."""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # File paths
    compressed_file = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_file = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    # Check if already exists
    if extracted_file.exists():
        print("✅ Lip sync model already exists!")
        return True
    
    # Download URL
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    print("Downloading lip sync model...")
    print(f"URL: {url}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, compressed_file)
        print("Download complete!")
        
        # Extract the file
        print("Extracting...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        compressed_file.unlink()
        
        print("✅ Lip sync model installed successfully!")
        print(f"Location: {extracted_file.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading lip sync model: {e}")
        print("\nYou can manually download from:")
        print(f"1. Download: {url}")
        print(f"2. Extract and place at: {extracted_file.absolute()}")
        return False

if __name__ == "__main__":
    download_and_extract() 