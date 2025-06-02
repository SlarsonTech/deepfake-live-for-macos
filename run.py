#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
from typing import Optional
import threading
import queue
import time

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import modules
from modules.core import FaceSwapper
from modules.ui import DeepfakeApp
from modules.camera import CameraProcessor
from modules.utils import check_requirements, download_models, optimize_for_macos

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deepfake Live Camera for macOS')
    parser.add_argument(
        '--execution-provider', 
        type=str, 
        default='coreml',
        choices=['cpu', 'coreml'],
        help='Execution provider for inference'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        help='Path to source face image'
    )
    parser.add_argument(
        '--camera-id', 
        type=int, 
        default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--no-gui', 
        action='store_true',
        help='Run without GUI (CLI mode)'
    )
    parser.add_argument(
        '--performance', 
        type=str, 
        default='fast',
        choices=['fast', 'balanced', 'quality'],
        help='Performance mode (default: fast)'
    )
    return parser.parse_args()

def main():
    """Main application entry point."""
    args = parse_args()
    
    # Apply macOS optimizations
    optimize_for_macos()
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        print("Missing requirements. Please install all dependencies.")
        sys.exit(1)
    
    # Download models if needed
    print("Checking models...")
    if not download_models():
        print("Failed to download models.")
        sys.exit(1)
    
    # Initialize face swapper
    print(f"Initializing face swapper with {args.execution_provider} provider...")
    face_swapper = FaceSwapper(execution_provider=args.execution_provider)
    
    # Initialize camera processor
    camera_processor = CameraProcessor(
        camera_id=args.camera_id,
        face_swapper=face_swapper
    )
    
    # Set default performance mode
    camera_processor.set_performance_mode(args.performance)
    print(f"Performance mode: {args.performance}")
    
    if args.no_gui:
        # CLI mode
        if not args.source:
            print("Error: --source argument is required in no-gui mode")
            sys.exit(1)
        
        print("Running in CLI mode...")
        camera_processor.set_source_image(args.source)
        camera_processor.start()
        
        print("Press 'q' to quit...")
        try:
            while True:
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        
        camera_processor.stop()
    else:
        # GUI mode
        print("Starting GUI...")
        app = DeepfakeApp(
            face_swapper=face_swapper,
            camera_processor=camera_processor
        )
        app.run()
    
    print("Application closed.")

if __name__ == "__main__":
    main() 