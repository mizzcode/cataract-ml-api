import tensorflow as tf
import numpy as np
import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = ['Cataract', 'Normal']  # Adjust order based on your training
        
        # Cloud storage URLs
        self.model_urls = [
            "https://drive.google.com/uc?export=download&id=1Ee95n98Oy6kLFO9sDXlTvgYsbedxe_Ww",
        ]
        
    def download_model_from_cloud(self, download_path: str):
        """Download model dari cloud storage"""
        logger.info("Model not found locally, attempting to download from cloud...")
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        for url in self.model_urls:
            try:
                logger.info(f"Downloading model from: {url}")
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                logger.info(f"Downloading model ({total_size / (1024*1024):.1f} MB)...")
                
                with open(download_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (1024*1024) == 0:  # Log setiap 1MB
                                    logger.info(f"Download progress: {progress:.1f}%")
                
                logger.info("Model downloaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"Failed to download from {url}: {e}")
                continue
        
        logger.error("Failed to download model from all URLs")
        return False
        
    def load_model(self, model_path: str = None):
        """Load the VGG16 .keras model"""
        try:
            # Coba cari model local dulu
            try:
                final_model_path = self._get_model_path(model_path)
            except FileNotFoundError:
                # Jika tidak ada, coba download
                logger.info("Model not found locally, attempting download...")
                
                # Set default download path
                download_path = "model/model_vgg16.keras"
                
                if self.download_model_from_cloud(download_path):
                    final_model_path = download_path
                else:
                    raise FileNotFoundError("Could not find or download model file")
            
            # Load model
            logger.info(f"Loading model from: {final_model_path}")
            self.model = tf.keras.models.load_model(final_model_path)
            self.model_path = final_model_path
            
            logger.info(f"VGG16 model loaded successfully from {final_model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
            # Print model summary
            self.model.summary()
            
            # Test the model
            self._test_model()
            
        except Exception as e:
            logger.error(f"Error loading VGG16 model: {str(e)}")
            raise e
    
    def _get_model_path(self, model_path: str = None) -> str:
        """Get the VGG16 .keras model path"""
        if model_path is not None and Path(model_path).exists():
            return model_path
            
        # Get current working directory and script directory
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent  # Go up to project root
        
        possible_paths = [
            # Relative to current working directory
            current_dir / "model_vgg16.keras",
            current_dir / "model" / "model_vgg16.keras",
            current_dir / "models" / "model_vgg16.keras",
            
            # Relative to script directory
            script_dir / "model_vgg16.keras",
            script_dir / "model" / "model_vgg16.keras",
            
            # Relative to project root
            project_root / "model_vgg16.keras",
            project_root / "model" / "model_vgg16.keras",
            project_root / "models" / "model_vgg16.keras",
            project_root / "Machine_Learning_API" / "model_vgg16.keras",
            project_root / "Machine_Learning_API" / "models" / "model_vgg16.keras",
            
            # Additional common locations
            Path("./model_vgg16.keras"),
            Path("../model_vgg16.keras"),
            Path("../../model_vgg16.keras"),
            Path("../../../model_vgg16.keras"),
        ]
        
        # Log current directory for debugging
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Project root: {project_root}")
        
        for path in possible_paths:
            abs_path = path.resolve()
            logger.info(f"Checking path: {abs_path}")
            if abs_path.exists():
                logger.info(f"Found VGG16 model at: {abs_path}")
                return str(abs_path)
        
        # List available files for debugging
        logger.error("Available files in current directory:")
        try:
            for item in current_dir.iterdir():
                logger.error(f"  {item}")
        except:
            logger.error("Cannot list current directory")
            
        logger.error("Available files in script directory:")
        try:
            for item in script_dir.iterdir():
                logger.error(f"  {item}")
        except:
            logger.error("Cannot list script directory")
            
        raise FileNotFoundError(f"model_vgg16.keras file not found. Searched in {len(possible_paths)} locations.")

    def _test_model(self):
        """Test the VGG16 model with dummy input"""
        try:
            # VGG16 typically expects (224, 224, 3) input
            input_shape = self.model.input_shape[1:]  # Remove batch dimension
            dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
            
            # VGG16 preprocessing: normalize to [0, 1] or use ImageNet preprocessing
            dummy_input = dummy_input * 255.0  # Scale to [0, 255] if needed
            
            # Test prediction
            prediction = self.model.predict(dummy_input, verbose=0)
            logger.info(f"VGG16 model test successful. Output shape: {prediction.shape}")
            logger.info(f"Sample prediction: {prediction}")
            
        except Exception as e:
            logger.warning(f"VGG16 model test failed: {e}")
    
    def predict(self, processed_image: np.ndarray):
        """Make prediction using VGG16 model with detailed logging"""
        if self.model is None:
            raise ValueError("VGG16 model not loaded. Call load_model() first.")
        
        try:
            # Log input details
            logger.info(f"Input image shape: {processed_image.shape}")
            logger.info(f"Input pixel range: {processed_image.min()} - {processed_image.max()}")
            
            # Ensure image has batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=1)
            
            logger.info(f"Raw VGG16 predictions shape: {predictions.shape}")
            logger.info(f"Raw VGG16 predictions: {predictions}")
            
            # Process VGG16 model predictions
            if predictions.shape[-1] == 1:
                # Binary classification with sigmoid output
                raw_confidence = float(predictions[0][0])
                logger.info(f"VGG16 sigmoid output: {raw_confidence}")
                
                # Determine predicted class
                predicted_class = 1 if raw_confidence > 0.5 else 0
                
                # Calculate confidence for the predicted class
                if predicted_class == 1:
                    final_confidence = raw_confidence
                else:
                    final_confidence = 1 - raw_confidence
                
                all_probs = [1 - raw_confidence, raw_confidence]
                
            elif predictions.shape[-1] == 2:
                # Binary classification with softmax output
                logger.info(f"VGG16 softmax outputs: {predictions[0]}")
                predicted_class = int(np.argmax(predictions[0]))
                final_confidence = float(np.max(predictions[0]))
                all_probs = predictions[0].tolist()
                
            else:
                # Multi-class classification
                predicted_class = int(np.argmax(predictions[0]))
                final_confidence = float(np.max(predictions[0]))
                all_probs = predictions[0].tolist()
            
            result = {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': final_confidence,
                'all_probabilities': all_probs,
                'raw_prediction': predictions[0].tolist(),
                'model_type': 'VGG16'
            }
            
            logger.info(f"VGG16 prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during VGG16 prediction: {str(e)}")
            raise e
    
    def get_model_info(self):
        """Get VGG16 model information"""
        if self.model is None:
            return {
                'loaded': False,
                'model_type': 'VGG16',
                'class_names': self.class_names
            }
        
        try:
            return {
                'loaded': True,
                'path': self.model_path,
                'model_type': 'VGG16 Keras',
                'class_names': self.class_names,
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'layers_count': len(self.model.layers),
                'last_layer_activation': self.model.layers[-1].activation.__name__ if hasattr(self.model.layers[-1], 'activation') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting VGG16 model info: {str(e)}")
            return {
                'loaded': True,
                'path': self.model_path,
                'model_type': 'VGG16 Keras',
                'class_names': self.class_names,
                'error': str(e)
            }

# Global VGG16 model instance
model_loader = ModelLoader()

def get_model():
    """Get the global VGG16 model instance"""
    if model_loader.model is None:
        model_loader.load_model()
    return model_loader