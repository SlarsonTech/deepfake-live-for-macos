"""
Camera processing module for real-time face swapping.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable

# Try to import lip sync module
try:
    from .lip_sync import LipSyncProcessor
    LIP_SYNC_AVAILABLE = True
except ImportError:
    LIP_SYNC_AVAILABLE = False
    print("Lip sync module not available. Running without lip sync support.")

class CameraProcessor:
    """Handles camera input and face swapping in real-time."""
    
    def __init__(self, camera_id: int = 0, face_swapper=None):
        """
        Initialize camera processor.
        
        Args:
            camera_id: Camera device ID
            face_swapper: FaceSwapper instance
        """
        self.camera_id = camera_id
        self.face_swapper = face_swapper
        
        # Camera capture
        self.cap = None
        self.is_running = False
        
        # Threading
        self.capture_thread = None
        self.process_thread = None
        
        # Frame queues
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        
        # Callbacks
        self.frame_callback = None
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance settings
        self.skip_frames = 2  # Process every Nth frame
        self.frame_counter = 0
        self.resolution_scale = 0.5  # Scale down for processing
        
        # Lip sync
        if LIP_SYNC_AVAILABLE:
            self.lip_sync = LipSyncProcessor()
            self.enable_lip_sync = True
        else:
            self.lip_sync = None
            self.enable_lip_sync = False
        
        # Store original frame for lip sync
        self.original_frame = None
        
    def set_source_image(self, image_path: str) -> bool:
        """
        Set source image for face swapping.
        
        Args:
            image_path: Path to source image
            
        Returns:
            True if successful
        """
        if self.face_swapper:
            return self.face_swapper.set_source_face(image_path)
        return False
    
    def set_frame_callback(self, callback: Callable):
        """Set callback for processed frames."""
        self.frame_callback = callback
    
    def start(self) -> bool:
        """Start camera capture and processing."""
        if self.is_running:
            return True
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # macOS specific optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.process_thread = threading.Thread(target=self._process_loop)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        return True
    
    def stop(self):
        """Stop camera capture and processing."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queues
        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
    
    def _capture_loop(self):
        """Capture frames from camera."""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Drop old frames
                    if self.input_queue.full():
                        try:
                            self.input_queue.get_nowait()
                        except:
                            pass
                    
                    try:
                        self.input_queue.put_nowait(frame)
                    except:
                        pass
            else:
                time.sleep(0.01)
    
    def _process_loop(self):
        """Process frames for face swapping."""
        last_processed_frame = None
        
        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                self.original_frame = frame.copy()  # Store for lip sync
                
                # Frame skipping for performance
                self.frame_counter += 1
                
                if self.frame_counter % self.skip_frames == 0:
                    # Resize frame for processing
                    height, width = frame.shape[:2]
                    small_frame = cv2.resize(
                        frame, 
                        (int(width * self.resolution_scale), 
                         int(height * self.resolution_scale))
                    )
                    
                    # Process frame
                    if self.face_swapper:
                        processed_small = self.face_swapper.process_frame(small_frame)
                        # Resize back to original
                        processed_frame = cv2.resize(processed_small, (width, height))
                        
                        # Apply lip sync if enabled
                        if self.enable_lip_sync and self.lip_sync and self.lip_sync.enabled:
                            processed_frame = self.lip_sync.process_frame(
                                self.original_frame, 
                                processed_frame
                            )
                    else:
                        processed_frame = frame
                    
                    last_processed_frame = processed_frame
                else:
                    # Use last processed frame
                    processed_frame = last_processed_frame if last_processed_frame is not None else frame
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Add FPS overlay
                cv2.putText(
                    processed_frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Add performance mode indicator
                mode_text = f"Mode: {'Fast' if self.skip_frames > 2 else 'Balanced'}"
                cv2.putText(
                    processed_frame,
                    mode_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Send to callback
                if self.frame_callback:
                    self.frame_callback(processed_frame)
                
                # Store in output queue
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        pass
                
                try:
                    self.output_queue.put_nowait(processed_frame)
                except:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process loop: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest processed frame."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode.
        
        Args:
            mode: 'fast', 'balanced', or 'quality'
        """
        if mode == 'fast':
            self.skip_frames = 3
            self.resolution_scale = 0.4
        elif mode == 'balanced':
            self.skip_frames = 2
            self.resolution_scale = 0.5
        else:  # quality
            self.skip_frames = 1
            self.resolution_scale = 0.8
    
    def toggle_lip_sync(self):
        """Toggle lip sync on/off."""
        if LIP_SYNC_AVAILABLE and self.lip_sync:
            self.enable_lip_sync = not self.enable_lip_sync
            return self.enable_lip_sync
        return False 