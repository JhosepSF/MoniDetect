"""
Inference service orchestrating the full MoniDetect cacao diagnosis pipeline.
Steps:
1. File validation
2. YOLO segmentation (cacao isolation on pure white background)
3. 224x224 RGB conversion & MobileNetV2 preprocess_input
4. MobileNetV2 1280-dim feature extraction
5. Downstream model classification (StandardScaler + KNN Classifier or SVC)
6. Optional Grad-CAM / Attention map computation
"""
import logging
import numpy as np
from .model_service import ModelManager, ModelNotFoundError, CacaoNotDetectedError
from .image_service import ImageService
from .gradcam_service import GradCAMService

logger = logging.getLogger(__name__)

MODEL_HUMAN_NAMES = {
    'knn_classifier': 'K-Nearest Neighbors (KNN Classifier)',
    'knn': 'K-Nearest Neighbors (KNN Classifier)',
    'svc': 'Support Vector Classifier (SVC)',
    'xgb_classifier': 'XGBoost Classifier',
    'svr': 'Support Vector Regressor (SVR)',
    'knn_regressor': 'KNN Regressor',
    'xgb_regressor': 'XGBoost Regressor',
}


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

        # Step 1: Verify presence of required models
        status = manager.check_models_status()
        if not status['is_ready_for_inference']:
            missing_names = ", ".join(status['missing_required'])
            raise ModelNotFoundError(
                status['missing_required'],
                message=f"No se puede realizar el análisis. Faltan los siguientes modelos en 'models/': {missing_names}"
            )

        metadata = manager.get_metadata()
        winner_key = metadata.get('winner_model', 'svr')
        winner_label = metadata.get('winner_model_label', 'SVR')
        winner_name = MODEL_HUMAN_NAMES.get(winner_key, MODEL_HUMAN_NAMES.get(winner_label.lower(), winner_label))
        winner_approach = metadata.get('winner_approach', 'regression_threshold')
        threshold = float(metadata.get('decision_threshold', 0.43))

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

        # Step 6: Feed features into downstream winning model
        downstream_model = manager.get_downstream_model()

        # Check if the pipeline contains a regressor step or approach is regression_threshold
        is_regressor = (
            winner_approach == 'regression_threshold' or
            (hasattr(downstream_model, 'named_steps') and any('regressor' in s or 'svr' in s for s in downstream_model.named_steps))
        )

        if is_regressor:
            # Regression with thresholding (e.g. StandardScaler + SVR)
            raw_score = float(downstream_model.predict(features)[0])
            class_id = int(raw_score >= threshold)
            is_monilia = (class_id == 1)
            class_name = "Monilia" if is_monilia else "Sano"
            decision_score = raw_score
            decision_score_display = f"{raw_score:+.4f}"
            score_type = "raw_regression_output"
            score_label = f"Score de Regresión ({winner_label})"
            score_note = f"Salida continua del regresor {winner_label}. Umbral fijo = {threshold:.2f} (< {threshold:.2f} Sano, ≥ {threshold:.2f} Monilia). Distancia al umbral: {raw_score - threshold:+.4f}."
            score_minus_threshold = round(raw_score - threshold, 4)

        else:
            # Direct classification
            prediction = downstream_model.predict(features)
            class_id = int(prediction[0]) if hasattr(prediction, '__iter__') else int(prediction)
            is_monilia = (class_id == 1)
            class_name = "Monilia" if is_monilia else "Sano"
            score_minus_threshold = 0.0

            if hasattr(downstream_model, 'predict_proba'):
                probabilities = downstream_model.predict_proba(features)
                prob_monilia = float(probabilities[0, 1])
                prob_sano = float(probabilities[0, 0])
                decision_score = prob_monilia
                decision_score_display = f"{prob_monilia * 100:.1f}%"
                score_type = "class_probability"
                score_label = "Probabilidad estimada (Monilia)"
                score_note = f"Consenso de vecinos más cercanos en el espacio latente MobileNetV2 (Sano: {prob_sano*100:.1f}%, Monilia: {prob_monilia*100:.1f}%)."

            elif hasattr(downstream_model, 'decision_function'):
                decision_raw = downstream_model.decision_function(features)
                decision_score = float(decision_raw[0]) if hasattr(decision_raw, '__iter__') else float(decision_raw)
                decision_score_display = f"{decision_score:+.4f}"
                score_type = "decision_function"
                score_label = "Score de Decisión (SVC)"
                score_note = "Puntuación bruta del hiperplano de decisión del Support Vector Classifier."

            else:
                decision_score = float(class_id)
                decision_score_display = f"{decision_score:.4f}"
                score_type = "classification_output"
                score_label = "Salida del Clasificador"
                score_note = "Clasificación directa del modelo seleccionado."

        # Formulate contextual messages
        if is_monilia:
            user_message = "Se detectaron características compatibles con moniliasis en el fruto analizado."
            alert_type = "danger"
        else:
            user_message = "No se detectaron características compatibles con moniliasis en el fruto analizado."
            alert_type = "success"

        disclaimer = "MoniDetect es una herramienta de apoyo basada en visión artificial y no sustituye una evaluación agronómica especializada."

        # Step 7: Optional Grad-CAM / Attention visualization
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
                    logger.warning("Fallo en Grad-CAM finetuned, intentando con Extractor: %s", str(e))
            
            # Fallback to direct Class Activation Mapping via Extractor + Downstream model
            if gradcam_b64 is None:
                try:
                    gradcam_b64 = GradCAMService.compute_cam_from_extractor_and_downstream(
                        extractor_model=extractor_model,
                        downstream_model=downstream_model,
                        preprocessed_tensor=preprocessed_tensor,
                        segmented_pil_224=segmented_pil
                    )
                except Exception as e:
                    logger.error("Error al calcular mapa de atención Grad-CAM: %s", str(e), exc_info=True)

        # Step 8: Prepare in-memory base64 representations
        orig_b64 = ImageService.image_to_base64(orig_pil, image_format="JPEG", quality=88)
        segmented_b64 = ImageService.image_to_base64(segmented_pil, image_format="JPEG", quality=90)

        return {
            "success": True,
            "class_id": class_id,
            "class_name": class_name,
            "winner_model_name": winner_name,
            "winner_model_key": winner_key,
            "decision_score": round(decision_score, 4),
            "decision_score_display": decision_score_display,
            "score_label": score_label,
            "score_note": score_note,
            "score_type": score_type,
            "message": user_message,
            "alert_type": alert_type,
            "disclaimer": disclaimer,
            "original_image": orig_b64,
            "segmented_image": segmented_b64,
            "gradcam_image": gradcam_b64,
            "has_gradcam": bool(gradcam_b64 is not None),
        }
