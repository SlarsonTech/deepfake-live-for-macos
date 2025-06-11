import unittest
from pathlib import Path
import cv2
from modules.core import FaceSwapper
import insightface
import onnxruntime as ort
import os

class FaceDetectionTest(unittest.TestCase):
    def test_face_jpeg_contains_face(self):
        img_path = Path(__file__).resolve().parent / "Face.jpeg"
        if not img_path.exists():
            self.skipTest(f"Test image not found at {img_path}")
        print("ONNX Runtime version:", ort.__version__)
        try:
            swapper = FaceSwapper(execution_provider='cpu')
        except Exception as e:
            self.fail(f"Error initializing FaceSwapper: {e}")

        print("Providers in use:", swapper.providers)
        print("Available ORT providers:", ort.get_available_providers())
        print("InsightFace version:", insightface.__version__)
        model_root = getattr(swapper.face_app, "root", None)
        if model_root:
            print("Face analysis model root:", model_root)
        else:
            print("Face analysis model root attribute not found")
        model_path = os.path.join('models', 'inswapper_128.onnx')
        print("Face swap model path:", os.path.abspath(model_path))
        print("Model exists:", os.path.exists(model_path))

        img = cv2.imread(str(img_path))
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        print("Image shape:", img.shape)
        print("Image dtype:", img.dtype)

        faces = swapper.face_app.get(img)
        print(f"Detected {len(faces)} faces")
        for i, f in enumerate(faces):
            print(f"Face {i}: bbox={f.bbox}, score={f.det_score}")
        self.assertGreater(len(faces), 0, "No faces detected in Face.jpeg")

if __name__ == '__main__':
    unittest.main()
