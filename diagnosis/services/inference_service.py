"""
Inference service orchestrating the full MoniDetect cacao diagnosis pipeline.
Steps:
1. File validation
2. YOLO segmentation (cacao isolation on pure white background)
3. 224x224 RGB conversion & MobileNetV2 preprocess_input
4. MobileNetV2 1280-dim feature extraction
5. SVC Pipeline classification (StandardScaler + SVC)
6. Optional Grad-CAM computation
"""
import logging
import numpy as np
from .model_service import ModelManager, ModelNotFoundError, CacaoNotDetectedError
from .image_service import ImageService
from .gradcam_service import GradCAMService

logger = logging.getLogger(__name__)

class InferenceService:
    """Main orchestrator for cacao image inference."""

    @classmethod
    def predict_cacao(cls, image_file, include_gradcam: bool = False) -> dict:
        """
        Executes the end-to-end diagnosis pipeline on the given image file.

        Args:
            image_file: Django UploadedFile or file-like object.
            include_gradcam: Whether to generate Grad-CAM heatmap visualization.

        Returns:
            dict with prediction details, decision score, messages, and base64 images.
        """
        manager = ModelManager()

        # Step 1: Verify presence of the 3 required models
        status = manager.check_models_status()
        if not status['is_ready_for_inference']:
            missing_names = ", ".join(status['missing_required'])
            raise ModelNotFoundError(
                status['missing_required'],
                message=f"No se puede realizar el análisis. Faltan los siguientes modelos en 'models/': {missing_names}"
            )

        # Step 2: Validate image file format and integrity
        ImageService.validate_image_file(image_file)
        orig_pil = ImageService.file_to_pil(image_file)

        # Step 3: Load YOLO model & perform cacao segmentation
        yolo_model = manager.get_yolo_model()
        segmented_pil = ImageService.segment_cacao_yolo(orig_pil, yolo_model, conf=0.25)

        # Step 4: Preprocess segmented image for MobileNetV2 (224x224, RGB, preprocess_input)
        preprocessed_tensor = ImageService.preprocess_for_mobilenet(segmented_pil)

        # Step 5: Extract 1280 features via MobileNetV2 extractor
        extractor_model = manager.get_mobilenet_extractor()
        features = extractor_model.predict(preprocessed_tensor, verbose=0)

        # Validate feature shape (1, 1280)
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        elif len(features.shape) > 2:
            features = features.reshape(1, -1)

        if features.shape[-1] != 1280:
            logger.warning("El vector de características tiene dimensión %s (esperada: 1280)", features.shape)

        # Step 6: Feed features into SVC pipeline (contains StandardScaler + SVC)
        svc_pipeline = manager.get_svc_pipeline()
        prediction = svc_pipeline.predict(features)
        decision_raw = svc_pipeline.decision_function(features)

        # Extract prediction class (0: Sano, 1: Monilia)
        class_id = int(prediction[0]) if hasattr(prediction, '__iter__') else int(prediction)
        is_monilia = (class_id == 1)

        class_name = "Monilia" if is_monilia else "Sano"

        # Extract decision score
        if hasattr(decision_raw, '__iter__'):
            decision_score = float(decision_raw[0])
        else:
            decision_score = float(decision_raw)

        # Formulate contextual messages
        if is_monilia:
            user_message = "Se detectaron características compatibles con moniliasis en el fruto analizado."
            alert_type = "danger"
        else:
            user_message = "No se detectaron características compatibles con moniliasis en el fruto analizado."
            alert_type = "success"

        disclaimer = "MoniDetect es una herramienta de apoyo basada en visión artificial y no sustituye una evaluación agronómica especializada."

        # Step 7: Optional Grad-CAM visualization
        gradcam_b64 = None
        
        if include_gradcam:
            if status['is_gradcam_available']:
                try:
                    finetuned_model = manager.get_finetuned_model()
                    gradcam_b64 = GradCAMService.compute_gradcam_from_finetuned(
                        model=finetuned_model,
                        preprocessed_tensor=preprocessed_tensor,
                        segmented_pil_224=segmented_pil
                    )
                except Exception as e:
                    logger.warning("Fallo en Grad-CAM finetuned, intentando con Extractor+SVC: %s", str(e))
            
            # Fallback to direct Class Activation Mapping via Extractor + SVC
            if gradcam_b64 is None:
                try:
                    gradcam_b64 = GradCAMService.compute_cam_from_extractor_and_svc(
                        extractor_model=extractor_model,
                        svc_pipeline=svc_pipeline,
                        preprocessed_tensor=preprocessed_tensor,
                        segmented_pil_224=segmented_pil
                    )
                except Exception as e:
                    logger.error("Error al calcular Grad-CAM fallback: %s", str(e), exc_info=True)

        # Step 8: Prepare in-memory base64 representations
        orig_b64 = ImageService.image_to_base64(orig_pil, image_format="JPEG", quality=88)
        segmented_b64 = ImageService.image_to_base64(segmented_pil, image_format="JPEG", quality=90)

        return {
            "success": True,
            "class_id": class_id,
            "class_name": class_name,
            "decision_score": round(decision_score, 4),
            "decision_score_display": f"{decision_score:+.4f}",
            "message": user_message,
            "alert_type": alert_type,
            "disclaimer": disclaimer,
            "original_image": orig_b64,
            "segmented_image": segmented_b64,
            "gradcam_image": gradcam_b64,
            "has_gradcam": bool(gradcam_b64 is not None),
        }
