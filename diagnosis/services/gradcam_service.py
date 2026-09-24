"""
Grad-CAM and Class Activation Mapping (CAM) service for MoniDetect.
Computes activation heatmaps on CNN feature representations to explain
which regions of the cacao fruit contributed most to the model representation.
Supports both:
1. Fine-tuned CNN model (mobilenetv2_segmented_final_finetuned.keras)
2. Direct Feature Extractor + Linear SVC Pipeline (mobilenetv2_segmented_final_extractor.keras + segmented_svc_final.joblib)
"""
import io
import base64
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAMService:
    """Service to compute and render CNN attention heatmaps."""

    @staticmethod
    def _find_target_conv_layer(model):
        """Locates the last 4D convolutional feature map layer in a model."""
        for layer in reversed(model.layers):
            if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
                return layer.name
            if 'conv' in layer.name.lower() or 'out_relu' in layer.name.lower():
                return layer.name
        return 'Conv_1'

    @classmethod
    def compute_gradcam_from_finetuned(
        cls,
        model,
        preprocessed_tensor: np.ndarray,
        segmented_pil_224: Image.Image,
        alpha: float = 0.45
    ) -> str:
        """Computes Grad-CAM using the complete fine-tuned CNN."""
        import tensorflow as tf
        import cv2

        target_layer_name = cls._find_target_conv_layer(model)
        logger.info("Generando Grad-CAM desde modelo finetuned (capa: %s)", target_layer_name)

        try:
            target_layer = model.get_layer(target_layer_name)
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[target_layer.output, model.output]
            )

            tensor_input = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(tensor_input)
                if predictions.shape[-1] == 1:
                    loss = predictions[0][0]
                else:
                    top_class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, top_class_idx]

            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_val = conv_outputs[0].numpy()
            pooled_grads_val = pooled_grads.numpy()

            for i in range(pooled_grads_val.shape[-1]):
                conv_outputs_val[:, :, i] *= pooled_grads_val[i]

            heatmap = np.mean(conv_outputs_val, axis=-1)
            heatmap = np.maximum(heatmap, 0)

            return cls._overlay_heatmap_on_image(heatmap, segmented_pil_224, alpha)

        except Exception as e:
            logger.error("Error al generar Grad-CAM en finetuned: %s", str(e), exc_info=True)
            return None

    @classmethod
    def compute_cam_from_extractor_and_svc(
        cls,
        extractor_model,
        svc_pipeline,
        preprocessed_tensor: np.ndarray,
        segmented_pil_224: Image.Image,
        alpha: float = 0.45
    ) -> str:
        """
        Computes the Class Activation Map (CAM) directly from the MobileNetV2
        feature extractor convolutional maps and the Linear SVC hyperplane weights.
        """
        import tensorflow as tf

        try:
            # Locate base model inside extractor
            if hasattr(extractor_model, 'get_layer') and 'mobilenetv2_1.00_224' in [l.name for l in extractor_model.layers]:
                base_model = extractor_model.get_layer('mobilenetv2_1.00_224')
            else:
                base_model = extractor_model

            # Locate last conv layer
            if 'out_relu' in [l.name for l in base_model.layers]:
                conv_layer = base_model.get_layer('out_relu')
            else:
                conv_layer_name = cls._find_target_conv_layer(base_model)
                conv_layer = base_model.get_layer(conv_layer_name)

            # Build sub-model to extract 4D feature map (e.g. 7x7x1280)
            feat_submodel = tf.keras.Model(inputs=base_model.input, outputs=conv_layer.output)
            tensor_input = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)
            fmap = feat_submodel(tensor_input, training=False)[0].numpy()  # (7, 7, 1280)

            # Extract weights from SVC linear classifier
            clf = svc_pipeline.named_steps.get('classifier', svc_pipeline.named_steps.get('svc'))
            scaler = svc_pipeline.named_steps.get('scaler', svc_pipeline.named_steps.get('standardscaler'))

            if hasattr(clf, 'coef_'):
                coef = clf.coef_[0]  # (1280,)
                if scaler is not None and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    weights = coef / (scaler.scale_ + 1e-7)
                else:
                    weights = coef
            else:
                weights = np.ones(fmap.shape[-1], dtype=np.float32)

            # Project feature map along classifier weights -> CAM
            cam = np.dot(fmap, weights)
            cam = np.maximum(cam, 0)  # ReLU

            return cls._overlay_heatmap_on_image(cam, segmented_pil_224, alpha)

        except Exception as e:
            logger.error("Error al calcular CAM desde Extractor + SVC: %s", str(e), exc_info=True)
            return None

    @staticmethod
    def _overlay_heatmap_on_image(heatmap: np.ndarray, segmented_pil_224: Image.Image, alpha: float = 0.45) -> str:
        """Normalizes heatmap, applies Jet colormap, blends with segmented image, and encodes to base64."""
        import cv2

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap_norm = heatmap / max_val
        else:
            heatmap_norm = np.zeros_like(heatmap)

        # Resize heatmap to 224x224
        heatmap_resized = cv2.resize(heatmap_norm, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply Jet colormap
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Base segmented cacao image
        orig_resized_rgb = np.array(segmented_pil_224.convert('RGB').resize((224, 224)))

        # Overlay blend
        overlay = cv2.addWeighted(colormap_rgb, alpha, orig_resized_rgb, 1.0 - alpha, 0)
        overlay_pil = Image.fromarray(overlay)

        # Encode to Base64
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="JPEG", quality=92)
        buffer.seek(0)
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
