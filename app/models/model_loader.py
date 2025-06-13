import tensorflow as tf
import numpy as np
import os
import requests
from pathlib import Path
import logging
import time

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
        """Debug function to understand the current environment"""
        logger.info("=== ENVIRONMENT DEBUG INFO ===")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python executable: {os.sys.executable}")
        logger.info(f"Environment variables:")
        for key in ['PWD', 'HOME', 'PATH']:
            logger.info(f"  {key}: {os.environ.get(key, 'Not set')}")
        
        # List all files and directories recursively from current directory
        logger.info("=== FILE SYSTEM STRUCTURE ===")
        current_dir = Path.cwd()
        try:
            for root, dirs, files in os.walk(current_dir):
                level = root.replace(str(current_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = Path(root) / file
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    logger.info(f"{subindent}{file} ({file_size} bytes)")
                    
                # Only go 3 levels deep to avoid too much output
                if level >= 2:
                    dirs.clear()
                    
        except Exception as e:
            logger.error(f"Error listing directory structure: {e}")
            
        logger.info("=== END DEBUG INFO ===")
        
    def download_model_from_cloud(self, download_path: str, max_retries: int = 3):
        """Enhanced download with better error handling and retries"""
        logger.info("Model not found locally, attempting to download from cloud...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        for url_index, url in enumerate(self.model_urls):
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempt {attempt + 1}/{max_retries} - Downloading from URL {url_index + 1}: {url[:100]}...")
                    
                    # Set headers to avoid blocking
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(url, stream=True, timeout=600, headers=headers)
                    response.raise_for_status()
                    
                    # Check if we got redirected to an error page
                    if 'content-length' not in response.headers:
                        logger.warning("No content-length header, might be an error page")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    logger.info(f"Starting download - Size: {total_size / (1024*1024):.1f} MB")
                    
                    # Download with progress tracking
                    downloaded = 0
                    start_time = time.time()
                    
                    with open(download_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Log progress every 10MB
                                if downloaded % (10 * 1024 * 1024) == 0:
                                    elapsed = time.time() - start_time
                                    speed = downloaded / elapsed / (1024 * 1024)  # MB/s
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        eta = (total_size - downloaded) / (downloaded / elapsed) if downloaded > 0 else 0
                                        logger.info(f"Progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) - Speed: {speed:.1f} MB/s - ETA: {eta:.0f}s")
                                    else:
                                        logger.info(f"Downloaded: {downloaded/(1024*1024):.1f} MB - Speed: {speed:.1f} MB/s")
                    
                    # Verify the downloaded file
                    if os.path.exists(download_path):
                        file_size = os.path.getsize(download_path)
                        logger.info(f"Download completed! File size: {file_size / (1024*1024):.1f} MB")
                        
                        # Basic validation - check if it's a valid file
                        if file_size < 1024:  # Less than 1KB is suspicious
                            logger.warning(f"Downloaded file is suspiciously small ({file_size} bytes)")
                            with open(download_path, 'r') as f:
                                content = f.read(500)  # Read first 500 chars
                                logger.warning(f"File content preview: {content}")
                            continue
                        
                        # Try to validate it's a Keras model
                        try:
                            # Quick check if it's a zip file (Keras models are zip files)
                            import zipfile
                            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                                file_list = zip_ref.namelist()
                                logger.info(f"Model file contains: {file_list[:5]}...")  # Show first 5 files
                                if 'saved_model.pb' in file_list or 'model.json' in file_list:
                                    logger.info("File appears to be a valid Keras model")
                                    return True
                        except Exception as e:
                            logger.warning(f"Could not validate model file: {e}")
                            # Continue anyway, might still work
                            
                        return True
                    else:
                        logger.error("File was not created after download")
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Timeout on attempt {attempt + 1} for URL {url_index + 1}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error on attempt {attempt + 1} for URL {url_index + 1}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt + 1} for URL {url_index + 1}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
        
        logger.error("Failed to download model from all URLs after all retries")
        return False
        
    def load_model(self, model_path: str = None):
        """Enhanced model loading with comprehensive debugging"""
        try:
            # Run debug first
            self.debug_environment()
            
            # Try to find model locally first
            try:
                final_model_path = self._get_model_path(model_path)
                logger.info(f"Found local model at: {final_model_path}")
            except FileNotFoundError as e:
                logger.info(f"Local model not found: {e}")
                
                # Set default download paths to try
                download_paths = [
                    "model/model_vgg16.keras",
                    "model_vgg16.keras",
                    "/app/model/model_vgg16.keras",  # Railway specific
                    "/tmp/model_vgg16.keras"  # Temporary directory
                ]
                
                download_success = False
                for download_path in download_paths:
                    logger.info(f"Attempting to download to: {download_path}")
                    try:
                        if self.download_model_from_cloud(download_path):
                            final_model_path = download_path
                            download_success = True
                            break
                    except Exception as download_error:
                        logger.error(f"Download to {download_path} failed: {download_error}")
                        continue
                
                if not download_success:
                    raise FileNotFoundError("Could not find or download model file from any source")
            
            # Load the model
            logger.info(f"Attempting to load model from: {final_model_path}")
            
            # Verify file exists and has reasonable size
            if not os.path.exists(final_model_path):
                raise FileNotFoundError(f"Model file does not exist: {final_model_path}")
                
            file_size = os.path.getsize(final_model_path)
            logger.info(f"Model file size: {file_size / (1024*1024):.1f} MB")
            
            if file_size < 1024:  # Less than 1KB
                raise ValueError(f"Model file is too small ({file_size} bytes), likely corrupted")
            
            # Load the model with error handling
            try:
                logger.info("Loading TensorFlow model...")
                self.model = tf.keras.models.load_model(final_model_path)
                self.model_path = final_model_path
                
                logger.info(f"✅ VGG16 model loaded successfully!")
                logger.info(f"Model input shape: {self.model.input_shape}")
                logger.info(f"Model output shape: {self.model.output_shape}")
                logger.info(f"Total parameters: {self.model.count_params():,}")
                
                # Test the model
                self._test_model()
                
            except Exception as load_error:
                logger.error(f"TensorFlow model loading failed: {load_error}")
                
                # Try to diagnose the issue
                try:
                    import zipfile
                    with zipfile.ZipFile(final_model_path, 'r') as zip_ref:
                        logger.info("Model file is a valid zip archive")
                        logger.info(f"Contents: {zip_ref.namelist()[:10]}")  # First 10 files
                except Exception as zip_error:
                    logger.error(f"Model file is not a valid zip archive: {zip_error}")
                
                raise load_error
                
        except Exception as e:
            logger.error(f"❌ Error loading VGG16 model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def _get_model_path(self, model_path: str = None) -> str:
        """Enhanced model path detection with better logging"""
        if model_path is not None and Path(model_path).exists():
            logger.info(f"Using provided model path: {model_path}")
            return model_path
            
        # Get various directory references
        current_dir = Path.cwd()
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        # More comprehensive path list
        possible_paths = [
            # Current directory variations
            current_dir / "model_vgg16.keras",
            current_dir / "model" / "model_vgg16.keras",
            current_dir / "app" / "model" / "model_vgg16.keras",
            
            # Script directory variations
            script_dir / "model_vgg16.keras",
            script_dir / "model" / "model_vgg16.keras",
            script_dir / ".." / "model_vgg16.keras",
            script_dir / ".." / "model" / "model_vgg16.keras",
            
            # Project root variations
            project_root / "model_vgg16.keras",
            project_root / "model" / "model_vgg16.keras",
            
            # Railway/Docker specific paths
            Path("/app/model_vgg16.keras"),
            Path("/app/model/model_vgg16.keras"),
            
            # Common relative paths
            Path("./model_vgg16.keras"),
            Path("../model_vgg16.keras"),
            Path("../../model_vgg16.keras"),
            Path("../../../model_vgg16.keras"),
        ]
        
        logger.info(f"🔍 Searching for model in {len(possible_paths)} locations...")
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Project root: {project_root}")
        
        for i, path in enumerate(possible_paths):
            try:
                abs_path = path.resolve()
                logger.info(f"  [{i+1:2d}] Checking: {abs_path}")
                
                if abs_path.exists():
                    file_size = abs_path.stat().st_size
                    logger.info(f"  ✅ FOUND! Size: {file_size / (1024*1024):.1f} MB")
                    return str(abs_path)
                else:
                    logger.info(f"  ❌ Not found")
                    
            except Exception as e:
                logger.warning(f"  ⚠️  Error checking path: {e}")
        
        # Enhanced directory listing for debugging
        logger.error("📁 DETAILED DIRECTORY ANALYSIS:")
        
        directories_to_check = [current_dir, script_dir, project_root, Path("/app")]
        
        for check_dir in directories_to_check:
            if check_dir.exists():
                logger.error(f"\n📂 Contents of {check_dir}:")
                try:
                    items = list(check_dir.iterdir())
                    for item in sorted(items):
                        if item.is_file():
                            size = item.stat().st_size
                            logger.error(f"  📄 {item.name} ({size} bytes)")
                        elif item.is_dir():
                            logger.error(f"  📁 {item.name}/")
                            # Check one level deep for model files
                            try:
                                sub_items = list(item.iterdir())
                                for sub_item in sorted(sub_items):
                                    if sub_item.is_file() and 'model' in sub_item.name.lower():
                                        sub_size = sub_item.stat().st_size
                                        logger.error(f"    📄 {sub_item.name} ({sub_size} bytes)")
                            except:
                                pass
                except Exception as e:
                    logger.error(f"  Error listing directory: {e}")
            else:
                logger.error(f"📂 {check_dir} does not exist")
            
        raise FileNotFoundError(f"model_vgg16.keras file not found in any of {len(possible_paths)} locations")

    def _test_model(self):
        """Enhanced model testing"""
        try:
            logger.info("🧪 Testing model with dummy input...")
            
            input_shape = self.model.input_shape[1:]  # Remove batch dimension
            logger.info(f"Expected input shape: {input_shape}")
            
            # Create dummy input
            dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
            dummy_input = dummy_input * 255.0  # Scale to [0, 255]
            
            logger.info(f"Test input shape: {dummy_input.shape}")
            logger.info(f"Test input range: {dummy_input.min():.2f} - {dummy_input.max():.2f}")
            
            # Test prediction
            start_time = time.time()
            prediction = self.model.predict(dummy_input, verbose=0)
            prediction_time = time.time() - start_time
            
            logger.info(f"✅ Model test successful!")
            logger.info(f"Prediction time: {prediction_time:.3f} seconds")
            logger.info(f"Output shape: {prediction.shape}")
            logger.info(f"Sample prediction: {prediction}")
            
            # Analyze output
            if prediction.shape[-1] == 1:
                logger.info("Model appears to use sigmoid activation (binary classification)")
            elif prediction.shape[-1] == 2:
                logger.info("Model appears to use softmax activation (binary classification)")
            else:
                logger.info(f"Model has {prediction.shape[-1]} outputs (multi-class)")
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            raise e
    
    def predict(self, processed_image: np.ndarray):
        """Enhanced prediction with detailed logging"""
        if self.model is None:
            raise ValueError("VGG16 model not loaded. Call load_model() first.")
        
        try:
            # Enhanced input validation and logging
            logger.info(f"🔮 Making prediction...")
            logger.info(f"Input shape: {processed_image.shape}")
            logger.info(f"Input dtype: {processed_image.dtype}")
            logger.info(f"Input range: {processed_image.min():.3f} - {processed_image.max():.3f}")
            logger.info(f"Input mean: {processed_image.mean():.3f}")
            logger.info(f"Input std: {processed_image.std():.3f}")
            
            # Ensure batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
                logger.info(f"Added batch dimension: {processed_image.shape}")
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(processed_image, verbose=1)
            prediction_time = time.time() - start_time
            
            logger.info(f"✅ Prediction completed in {prediction_time:.3f} seconds")
            logger.info(f"Raw predictions shape: {predictions.shape}")
            logger.info(f"Raw predictions: {predictions}")
            logger.info(f"Raw predictions sum: {predictions.sum():.6f}")
            
            # Enhanced prediction processing
            if predictions.shape[-1] == 1:
                # Binary classification with sigmoid
                raw_confidence = float(predictions[0][0])
                logger.info(f"Sigmoid output: {raw_confidence:.6f}")
                
                predicted_class = 1 if raw_confidence > 0.5 else 0
                final_confidence = raw_confidence if predicted_class == 1 else (1 - raw_confidence)
                all_probs = [1 - raw_confidence, raw_confidence]
                
            elif predictions.shape[-1] == 2:
                # Binary classification with softmax
                logger.info(f"Softmax outputs: {predictions[0]}")
                predicted_class = int(np.argmax(predictions[0]))
                final_confidence = float(np.max(predictions[0]))
                all_probs = predictions[0].tolist()
                
            else:
                # Multi-class
                predicted_class = int(np.argmax(predictions[0]))
                final_confidence = float(np.max(predictions[0]))
                all_probs = predictions[0].tolist()
            
            result = {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': final_confidence,
                'all_probabilities': all_probs,
                'raw_prediction': predictions[0].tolist(),
                'model_type': 'VGG16',
                'prediction_time': prediction_time
            }
            
            logger.info(f"🎯 Final result: {result['class_name']} ({result['confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def get_model_info(self):
        """Enhanced model information"""
        base_info = {
            'loaded': self.model is not None,
            'model_type': 'VGG16',
            'class_names': self.class_names,
            'model_urls': len(self.model_urls)
        }
        
        if self.model is None:
            return base_info
        
        try:
            return {
                **base_info,
                'path': self.model_path,
                'model_type': 'VGG16 Keras',
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
                'layers_count': len(self.model.layers),
                'optimizer': str(self.model.optimizer.__class__.__name__) if hasattr(self.model, 'optimizer') else 'unknown',
                'loss': str(self.model.loss) if hasattr(self.model, 'loss') else 'unknown',
                'last_layer_activation': self.model.layers[-1].activation.__name__ if hasattr(self.model.layers[-1], 'activation') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {
                **base_info,
                'path': self.model_path,
                'error': str(e)
            }

# Global model instance
model_loader = ModelLoader()

def get_model():
    """Get the global VGG16 model instance"""
    if model_loader.model is None:
        model_loader.load_model()
    return model_loader