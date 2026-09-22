"""
Model management service for MoniDetect.
Centralizes loading, caching, and health status for all AI models.
"""
import os
import threading
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    """Raised when one or more required model files are not found on disk."""
    def __init__(self, missing_models, message=None):
        self.missing_models = missing_models
        if message is None:
            models_str = ", ".join(missing_models)
            message = f"Faltan los siguientes archivos de modelos requeridos en la carpeta 'models/': {models_str}."
        super().__init__(message)


class ModelLoadError(Exception):
    """Raised when a model file exists but fails to load."""
    pass


class CacaoNotDetectedError(Exception):
    """Raised when YOLO segmentation cannot detect any cacao fruit in the image."""
    pass


class ModelManager:
    """
    Thread-safe Singleton class to manage the lifecycle of machine learning models.
    Loads models into memory once and caches them across requests.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self.models_dir = getattr(settings, 'MODELS_DIR', Path(settings.BASE_DIR) / 'models')
        self.model_filenames = getattr(settings, 'MODEL_FILES', {
            'yolo': 'Segmentador_Cacao_YOLO26n_best.pt',
            'extractor': 'mobilenetv2_segmented_final_extractor.keras',
            'svc': 'segmented_svc_final.joblib',
            'finetuned': 'mobilenetv2_segmented_final_finetuned.keras',
        })

        # Cache references for loaded models
        self._yolo_model = None
        self._mobilenet_extractor = None
        self._svc_pipeline = None
        self._finetuned_model = None

        # Lock for lazy model loading
        self._load_lock = threading.Lock()
        self._initialized = True

    def get_model_path(self, model_key: str) -> Path:
        """Returns the absolute Path for a given model key."""
        filename = self.model_filenames.get(model_key)
        if not filename:
            raise ValueError(f"Clave de modelo desconocida: '{model_key}'")
        return self.models_dir / filename

    def model_exists(self, model_key: str) -> bool:
        """Checks whether the model file physically exists on disk."""
        return self.get_model_path(model_key).is_file()

    def check_models_status(self) -> dict:
        """
        Inspects the 'models/' folder and returns the availability status
        of all required and optional model files.
        """
        required_keys = ['yolo', 'extractor', 'svc']
        optional_keys = ['finetuned']

        status = {
            'models_dir': str(self.models_dir),
            'files': {},
            'missing_required': [],
            'missing_optional': [],
            'is_ready_for_inference': True,
            'is_gradcam_available': True,
        }

        for key, filename in self.model_filenames.items():
            path = self.models_dir / filename
            exists = path.is_file()
            file_size = path.stat().st_size if exists else 0
            
            status['files'][key] = {
                'filename': filename,
                'path': str(path),
                'exists': exists,
                'size_bytes': file_size,
                'is_required': key in required_keys,
            }

            if not exists:
                if key in required_keys:
                    status['missing_required'].append(filename)
                    status['is_ready_for_inference'] = False
                elif key in optional_keys:
                    status['missing_optional'].append(filename)
                    status['is_gradcam_available'] = False

        return status

    def get_yolo_model(self):
        """
        Loads and returns the Ultralytics YOLO cacao segmentation model.
        Loads lazily and caches in memory.
        """
        if self._yolo_model is None:
            with self._load_lock:
                if self._yolo_model is None:
                    path = self.get_model_path('yolo')
                    if not path.is_file():
                        raise ModelNotFoundError([path.name])
                    try:
                        logger.info("Cargando modelo YOLO de segmentación desde: %s", path)
                        from ultralytics import YOLO
                        self._yolo_model = YOLO(str(path))
                    except Exception as e:
                        logger.error("Error al cargar modelo YOLO: %s", str(e), exc_info=True)
                        raise ModelLoadError(f"Error al inicializar el modelo YOLO ({path.name}): {str(e)}") from e
        return self._yolo_model

    def get_mobilenet_extractor(self):
        """
        Loads and returns the MobileNetV2 feature extractor (.keras).
        Loads lazily and caches in memory.
        """
        if self._mobilenet_extractor is None:
            with self._load_lock:
                if self._mobilenet_extractor is None:
                    path = self.get_model_path('extractor')
                    if not path.is_file():
                        raise ModelNotFoundError([path.name])
                    try:
                        logger.info("Cargando extractor MobileNetV2 desde: %s", path)
                        import tensorflow as tf
                        self._mobilenet_extractor = tf.keras.models.load_model(str(path), compile=False)
                    except Exception as e:
                        logger.error("Error al cargar extractor MobileNetV2: %s", str(e), exc_info=True)
                        raise ModelLoadError(f"Error al inicializar el extractor MobileNetV2 ({path.name}): {str(e)}") from e
        return self._mobilenet_extractor

    def get_svc_pipeline(self):
        """
        Loads and returns the StandardScaler + SVC pipeline (.joblib).
        Loads lazily and caches in memory.
        """
        if self._svc_pipeline is None:
            with self._load_lock:
                if self._svc_pipeline is None:
                    path = self.get_model_path('svc')
                    if not path.is_file():
                        raise ModelNotFoundError([path.name])
                    try:
                        logger.info("Cargando pipeline SVC desde: %s", path)
                        import joblib
                        self._svc_pipeline = joblib.load(str(path))
                    except Exception as e:
                        logger.error("Error al cargar pipeline SVC: %s", str(e), exc_info=True)
                        raise ModelLoadError(f"Error al inicializar el pipeline SVC ({path.name}): {str(e)}") from e
        return self._svc_pipeline

    def get_finetuned_model(self):
        """
        Loads and returns the fine-tuned CNN model for Grad-CAM.
        Loads lazily and caches in memory.
        """
        if self._finetuned_model is None:
            with self._load_lock:
                if self._finetuned_model is None:
                    path = self.get_model_path('finetuned')
                    if not path.is_file():
                        raise ModelNotFoundError([path.name])
                    try:
                        logger.info("Cargando modelo finetuned para Grad-CAM desde: %s", path)
                        import tensorflow as tf
                        self._finetuned_model = tf.keras.models.load_model(str(path), compile=False)
                    except Exception as e:
                        logger.error("Error al cargar modelo finetuned: %s", str(e), exc_info=True)
                        raise ModelLoadError(f"Error al inicializar el modelo finetuned ({path.name}): {str(e)}") from e
        return self._finetuned_model

    def clear_cache(self):
        """Resets loaded model references from memory (useful for tests or reload)."""
        with self._load_lock:
            self._yolo_model = None
            self._mobilenet_extractor = None
            self._svc_pipeline = None
            self._finetuned_model = None
            logger.info("Caché de modelos reiniciada.")
