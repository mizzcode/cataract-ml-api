import tensorflow as tf
import numpy as np
import os
import requests
from pathlib import Path
import logging
import time
import traceback

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = ['Cataract', 'Normal']
        self.model_urls = [
            "https://www.dropbox.com/scl/fi/myhukdszdsun9dtpdyekx/model_vgg16.keras?rlkey=kjjde07glo3oxb66uzbe76kv1&st=bivz28fr&dl=1",
            "https://drive.google.com/uc?export=download&id=1Ee95n98Oy6kLFO9sDXlTvgYsbedxe_Ww",
        ]

    def debug_environment(self):
        logger.info("=== ENVIRONMENT DEBUG INFO ===")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python executable: {os.sys.executable}")
        logger.info(f"Environment variables:")
        for key in ['PWD', 'HOME', 'PATH']:
            logger.info(f"  {key}: {os.environ.get(key, 'Not set')}")
        logger.info("=== END DEBUG INFO ===")

    def _get_model_path(self, model_path: str = None) -> str:
        if model_path is not None and Path(model_path).exists():
            logger.info(f"Using provided model path: {model_path}")
            return model_path

        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "model" / "model_vgg16.keras",  # Root level priority
            current_dir / "app" / "model" / "model_vgg16.keras",  # Fallback
            Path("/app/model/model_vgg16.keras"),  # Railway/Docker path
        ]

        logger.info(f"🔍 Searching for model in {len(possible_paths)} locations...")
        for i, path in enumerate(possible_paths):
            abs_path = path.resolve()
            logger.info(f"  [{i+1}] Checking: {abs_path}")
            if abs_path.exists():
                file_size = abs_path.stat().st_size
                logger.info(f"  ✅ FOUND! Size: {file_size / (1024*1024):.1f} MB")
                return str(abs_path)
            logger.info(f"  ❌ Not found")

        raise FileNotFoundError(f"model_vgg16.keras not found in any of {len(possible_paths)} locations")

    def load_model(self, model_path: str = None):
        try:
            self.debug_environment()

            final_model_path = "/app/model/model_vgg16.keras"
            logger.info(f"Found model at: {final_model_path}")

            file_size = os.path.getsize(final_model_path)
            logger.info(f"Model file size: {file_size / (1024*1024):.1f} MB")

            if file_size < 1024 * 1024:  # Less than 1MB (likely LFS pointer)
                logger.error(f"Model file {final_model_path} is too small ({file_size} bytes), likely an LFS pointer")
                raise ValueError(f"LFS download failed for {final_model_path}")

            self.model = tf.keras.models.load_model(final_model_path)
            self.model_path = final_model_path
            logger.info("✅ Model loaded successfully!")
            self._test_model()

        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _test_model(self):
        try:
            logger.info("🧪 Testing model with dummy input...")
            input_shape = self.model.input_shape[1:]
            dummy_input = np.random.random((1,) + input_shape).astype(np.float32) * 255.0
            self.model.predict(dummy_input, verbose=0)
            logger.info("✅ Model test successful!")
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            raise

    def predict(self, processed_image: np.ndarray):
        if self.model is None:
            raise ValueError("Model not loaded")
        try:
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            return {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'all_probabilities': predictions[0].tolist(),
            }
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise

    def get_model_info(self):
        if self.model is None:
            return {'loaded': False}
        return {
            'loaded': True,
            'path': self.model_path,
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
        }

model_loader = ModelLoader()

def get_model():
    if model_loader.model is None:
        model_loader.load_model()
    return model_loader