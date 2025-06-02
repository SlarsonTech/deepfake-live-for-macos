"""
Lip sync module for real-time mouth movement transfer.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import dlib

class LipSyncProcessor:
    """Handles lip sync by transferring mouth movements."""
    
    def __init__(self):
        """Initialize lip sync processor."""
        # Load facial landmark detector
        try:
            self.detector = dlib.get_frontal_face_detector()
            # You'll need to download this file
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.enabled = True
        except:
            print("Warning: Lip sync models not found. Lip sync will be disabled.")
            print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.enabled = False
        
        # Mouth landmark indices (points 48-67 in dlib's 68-point model)
        self.mouth_points = list(range(48, 68))
        
    def get_mouth_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mouth landmarks from image.
        
        Args:
            image: Input image
            
        Returns:
            Array of mouth landmark points or None
        """
        if not self.enabled:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Get landmarks for first face
        shape = self.predictor(gray, faces[0])
        
        # Extract mouth points
        mouth_points = []
        for i in self.mouth_points:
            mouth_points.append([shape.part(i).x, shape.part(i).y])
        
        return np.array(mouth_points, dtype=np.float32)
    
    def transfer_mouth(self, source_frame: np.ndarray, target_frame: np.ndarray, 
                      source_mouth: np.ndarray, target_mouth: np.ndarray) -> np.ndarray:
        """
        Transfer mouth from source to target frame.
        
        Args:
            source_frame: Source video frame
            target_frame: Target frame with swapped face
            source_mouth: Source mouth landmarks
            target_mouth: Target mouth landmarks
            
        Returns:
            Frame with transferred mouth
        """
        if source_mouth is None or target_mouth is None:
            return target_frame
        
        # Create mouth mask
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [target_mouth.astype(np.int32)], 255)
        
        # Get bounding boxes
        sx, sy, sw, sh = cv2.boundingRect(source_mouth.astype(np.int32))
        tx, ty, tw, th = cv2.boundingRect(target_mouth.astype(np.int32))
        
        # Add padding
        padding = 10
        sx = max(0, sx - padding)
        sy = max(0, sy - padding)
        sw = min(source_frame.shape[1] - sx, sw + 2 * padding)
        sh = min(source_frame.shape[0] - sy, sh + 2 * padding)
        
        tx = max(0, tx - padding)
        ty = max(0, ty - padding)
        tw = min(target_frame.shape[1] - tx, tw + 2 * padding)
        th = min(target_frame.shape[0] - ty, th + 2 * padding)
        
        try:
            # Extract mouth regions
            source_mouth_region = source_frame[sy:sy+sh, sx:sx+sw]
            
            # Resize source mouth to match target
            if source_mouth_region.size > 0:
                resized_mouth = cv2.resize(source_mouth_region, (tw, th))
                
                # Create result
                result = target_frame.copy()
                
                # Blend mouth region
                mouth_mask = mask[ty:ty+th, tx:tx+tw]
                mouth_mask_3channel = cv2.cvtColor(mouth_mask, cv2.COLOR_GRAY2BGR) / 255.0
                
                # Seamless cloning for better blending
                center = (tx + tw // 2, ty + th // 2)
                
                # Normal blend first
                result[ty:ty+th, tx:tx+tw] = (
                    result[ty:ty+th, tx:tx+tw] * (1 - mouth_mask_3channel) +
                    resized_mouth * mouth_mask_3channel
                ).astype(np.uint8)
                
                # Try seamless clone if available
                try:
                    result = cv2.seamlessClone(
                        resized_mouth,
                        result,
                        mouth_mask,
                        center,
                        cv2.NORMAL_CLONE
                    )
                except:
                    pass  # Fallback to normal blend if seamless clone fails
                
                return result
            
        except Exception as e:
            print(f"Error in mouth transfer: {e}")
        
        return target_frame
    
    def process_frame(self, source_frame: np.ndarray, swapped_frame: np.ndarray) -> np.ndarray:
        """
        Process frame with lip sync.
        
        Args:
            source_frame: Original camera frame
            swapped_frame: Frame with swapped face
            
        Returns:
            Frame with lip sync applied
        """
        if not self.enabled:
            return swapped_frame
        
        # Get mouth landmarks
        source_mouth = self.get_mouth_landmarks(source_frame)
        target_mouth = self.get_mouth_landmarks(swapped_frame)
        
        # Transfer mouth
        return self.transfer_mouth(source_frame, swapped_frame, source_mouth, target_mouth) 