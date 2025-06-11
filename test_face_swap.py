#!/usr/bin/env python3
"""Test script to debug face swapping issues."""

import cv2
from pathlib import Path
from modules.core import FaceSwapper
import insightface
import onnxruntime as ort
import os

def test_face_swap():
    print("=== Face Swap Test ===\n")
    
    # Initialize face swapper
    print("1. Initializing face swapper...")
    print("ONNX Runtime version:", ort.__version__)
    try:
        face_swapper = FaceSwapper(execution_provider='cpu')  # Use CPU for testing
    except Exception as e:
        print(f"Error initializing FaceSwapper: {e}")
        return False

    print("Providers in use:", face_swapper.providers)
    print("Available ORT providers:", ort.get_available_providers())
    print("InsightFace version:", insightface.__version__)
    model_root = getattr(face_swapper.face_app, "root", None)
    if model_root:
        print("Face analysis model root:", model_root)
    else:
        print("Face analysis model root attribute not found")
    model_path = os.path.join('models', 'inswapper_128.onnx')
    print("Face swap model path:", os.path.abspath(model_path))
    print("Model exists:", os.path.exists(model_path))
    
    # Check if model is loaded
    if face_swapper.face_swapper is None:
        print("\n❌ Face swap model is not loaded!")
        print("Please download the model as instructed above.")
        return False
    else:
        print("✅ Face swap model loaded successfully!")
    
    # Test face detection
    print("\n2. Testing face detection...")
    
    # Load the provided Face.jpeg for detection
    image_path = Path(__file__).resolve().parent / "tests" / "Face.jpeg"
    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found at {image_path}. "
            "Place a Face.jpeg image in the tests directory."
        )

    test_image = cv2.imread(str(image_path))
    if test_image is None:
        raise RuntimeError(f"Failed to read image at {image_path}")
    print("Image shape:", test_image.shape)
    print("Image dtype:", test_image.dtype)
    
    try:
        faces = face_swapper.face_app.get(test_image)
        print(f"✅ Face detection working. Found {len(faces)} faces.")
        for i, f in enumerate(faces):
            print(f"Face {i}: bbox={f.bbox}, score={f.det_score}")
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False
    
    print("\n3. Checking InsightFace models...")
    import os
    insightface_path = os.path.expanduser("~/.insightface/models")
    if os.path.exists(insightface_path):
        print(f"InsightFace models directory: {insightface_path}")
        for root, dirs, files in os.walk(insightface_path):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
    else:
        print("InsightFace models directory not found.")
    
    return True

if __name__ == "__main__":
    test_face_swap() 
