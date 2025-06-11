import unittest
from pathlib import Path
import cv2
from modules.core import FaceSwapper

class FaceDetectionTest(unittest.TestCase):
    def test_face_jpeg_contains_face(self):
        img_path = Path(__file__).resolve().parent / "Face.jpeg"
        if not img_path.exists():
            self.skipTest(f"Test image not found at {img_path}")
        swapper = FaceSwapper(execution_provider='cpu')
        img = cv2.imread(str(img_path))
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        faces = swapper.face_app.get(img)
        self.assertGreater(len(faces), 0, "No faces detected in Face.jpeg")

if __name__ == '__main__':
    unittest.main()
