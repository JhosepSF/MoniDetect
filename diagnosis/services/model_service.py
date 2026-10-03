"""
Model management service for MoniDetect.
Centralizes loading, caching, and health status for all AI models.
"""
import os
import json
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

    # Candidate filenames per model role in priority order
    MODEL_CANDIDATES = {
        'yolo': [
            'cacao_yolo_segmenter.pt',
            'Segmentador_Cacao_YOLO26n_best.pt',
        ],
        'extractor': [
            'mobilenetv2_feature_extractor.keras',
            'mobilenetv2_segmented_final_extractor.keras',
        ],
        'downstream': [
            'final_regressor.joblib',
            'final_downstream_model.joblib',
            'segmented_svr_final.joblib',
            'segmented_knn_final.joblib',
            'segmented_svc_final.joblib',
        ],
        'finetuned': [
            'mobilenetv2_segmented_final_finetuned.keras',
        ],
        'metadata': [
            'model_metadata.json',
            'final_segmented_pipeline_metadata.json',
        ]
    }

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

        # Cache references for loaded models
        self._yolo_model = None
        self._mobilenet_extractor = None
        self._downstream_model = None
        self._finetuned_model = None
        self._metadata = None

        # Lock for lazy model loading
        self._load_lock = threading.Lock()
        self._initialized = True

    def resolve_model_path(self, role: str) -> Path:
        """
        Resolves the physical Path for a model role by checking candidate filenames.
        Returns the first existing file or the primary filename path if none exist.
        """
        candidates = self.MODEL_CANDIDATES.get(role, [])
        for fname in candidates:
            candidate_path = self.models_dir / fname
            if candidate_path.is_file():
                return candidate_path
        
        # Default to primary candidate if none exist on disk
        primary_name = candidates[0] if candidates else f"{role}.model"
        return self.models_dir / primary_name

    def get_model_path(self, model_key: str) -> Path:
        """Returns the resolved absolute Path for a given model key."""
        if model_key in ('svc', 'knn', 'svr', 'classifier', 'regressor'):
            model_key = 'downstream'
        return self.resolve_model_path(model_key)

    def model_exists(self, model_key: str) -> bool:
        """Checks whether the model file physically exists on disk."""
        return self.get_model_path(model_key).is_file()

    def get_metadata(self) -> dict:
        """Loads and returns model_metadata.json if present."""
        if self._metadata is None:
            meta_path = self.resolve_model_path('metadata')
            if meta_path.is_file():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self._metadata = json.load(f)
                except Exception as e:
                    logger.warning("No se pudo leer model_metadata.json: %s", str(e))
                    self._metadata = {}
            else:
                self._metadata = {}
        return self._metadata

    def get_decision_threshold(self) -> float:
        """Returns the configured decision threshold for regression models."""
        meta = self.get_metadata()
        thresh = meta.get('decision_threshold')
        if thresh is not None:
            try:
                return float(thresh)
            except (ValueError, TypeError):
                pass
        return 0.43

    def check_models_status(self) -> dict:
        """
        Inspects the 'models/' folder and returns the availability status
        of all required and optional model files.
        """
        required_roles = ['yolo', 'extractor', 'downstream']
        optional_roles = ['finetuned', 'metadata']

        status = {
            'models_dir': str(self.models_dir),
            'files': {},
            'missing_required': [],
            'missing_optional': [],
            'is_ready_for_inference': True,
            'is_gradcam_available': True,
            'winner_model': None,
            'winner_model_label': None,
            'winner_approach': None,
            'decision_threshold': self.get_decision_threshold(),
        }

        # Check metadata
        meta = self.get_metadata()
        if meta:
            status['winner_model'] = meta.get('winner_model', 'svr')
            status['winner_model_label'] = meta.get('winner_model_label', 'SVR')
            status['winner_approach'] = meta.get('winner_approach', 'regression_threshold')
            status['decision_threshold'] = meta.get('decision_threshold', 0.43)

        for role in required_roles + optional_roles:
            path = self.resolve_model_path(role)
            exists = path.is_file()
            file_size = path.stat().st_size if exists else 0
            
            status['files'][role] = {
                'filename': path.name,
                'path': str(path),
                'exists': exists,
                'size_bytes': file_size,
                'is_required': role in required_roles,
            }

            if not exists:
                if role in required_roles:
                    status['missing_required'].append(path.name)
                    status['is_ready_for_inference'] = False
                elif role == 'finetuned':
                    status['missing_optional'].append(path.name)
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
                    path = self.resolve_model_path('yolo')
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
                    path = self.resolve_model_path('extractor')
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

    def get_downstream_model(self):
        """
        Loads and returns the final downstream classification/regression model pipeline (.joblib).
        Loads lazily and caches in memory.
        """
        if self._downstream_model is None:
            with self._load_lock:
                if self._downstream_model is None:
                    path = self.resolve_model_path('downstream')
                    if not path.is_file():
                        raise ModelNotFoundError([path.name])
                    try:
                        logger.info("Cargando modelo downstream ganador desde: %s", path)
                        import joblib
                        self._downstream_model = joblib.load(str(path))
                    except Exception as e:
                        logger.error("Error al cargar modelo downstream: %s", str(e), exc_info=True)
                        raise ModelLoadError(f"Error al inicializar el modelo downstream ({path.name}): {str(e)}") from e
        return self._downstream_model

    def get_svc_pipeline(self):
        """Alias for backward compatibility with older services."""
        return self.get_downstream_model()

    def get_finetuned_model(self):
        """
        Loads and returns the fine-tuned CNN model for Grad-CAM.
        Loads lazily and caches in memory.
        """
        if self._finetuned_model is None:
            with self._load_lock:
                if self._finetuned_model is None:
                    path = self.resolve_model_path('finetuned')
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
            self._downstream_model = None
            self._finetuned_model = None
            self._metadata = None
            logger.info("Caché de modelos reiniciada.")
