"""
Grad-CAM (Gradient-weighted Class Activation Mapping) service for MoniDetect.
Computes activation heatmaps on mobilenetv2_segmented_final_finetuned.keras
to explain CNN feature representations without modifying model weights.
"""
import io
import base64
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAMService:
    """Service to compute and render Grad-CAM attention heatmaps."""

    @staticmethod
    def _find_target_conv_layer(model):
        """
        Locates the last 4D convolutional feature map layer in the model graph.
        In MobileNetV2 architectures, this is typically 'Conv_1', 'out_relu',
        or the last layer producing a 4D tensor (batch, h, w, channels).
        """
        # Search from the last layer backwards
        for layer in reversed(model.layers):
            # Check for standard conv layer names or 4D output shapes
            if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
                return layer.name
            if 'conv' in layer.name.lower() or 'out_relu' in layer.name.lower():
                return layer.name

        # Fallback to standard MobileNetV2 final conv name
        return 'Conv_1'

    @classmethod
    def compute_gradcam(
        cls,
        model,
        preprocessed_tensor: np.ndarray,
        segmented_pil_224: Image.Image,
        target_layer_name: str = None,
        alpha: float = 0.45
    ) -> str:
        """
        Computes the Grad-CAM heatmap for the given preprocessed input tensor
        and overlays it onto the 224x224 segmented cacao image.
        
        Returns:
            Base64 data URI of the overlaid Grad-CAM visualization image.
        """
        import tensorflow as tf
        import cv2

        if target_layer_name is None:
            target_layer_name = cls._find_target_conv_layer(model)

        logger.info("Calculando Grad-CAM utilizando la capa convolucional: '%s'", target_layer_name)

        try:
            # Create a sub-model that outputs both the target conv feature map and model predictions
            target_layer = model.get_layer(target_layer_name)
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[target_layer.output, model.output]
            )

            tensor_input = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(tensor_input)
                # If output is sigmoid / binary (1 node) or categorical (2 nodes)
                if predictions.shape[-1] == 1:
                    loss = predictions[0][0]
                else:
                    top_class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, top_class_idx]

            # Compute gradients of top class score with respect to feature maps
            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                logger.warning("No se pudieron calcular gradientes para la capa %s", target_layer_name)
                return None

            # Global average pooling of gradients -> importance weights per feature map
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the feature maps by their gradient importance
            conv_outputs_val = conv_outputs[0].numpy()
            pooled_grads_val = pooled_grads.numpy()

            for i in range(pooled_grads_val.shape[-1]):
                conv_outputs_val[:, :, i] *= pooled_grads_val[i]

            # Heatmap is the mean across channel axis with ReLU (discard negative contributions)
            heatmap = np.mean(conv_outputs_val, axis=-1)
            heatmap = np.maximum(heatmap, 0)

            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            else:
                heatmap = np.zeros_like(heatmap)

            # Resize heatmap to 224x224 to match input image
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Apply Jet colormap
            colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

            # Prepare segmented image (224x224 RGB)
            orig_resized_rgb = np.array(segmented_pil_224.convert('RGB').resize((224, 224)))

            # Superimpose heatmap onto the segmented image
            overlay = cv2.addWeighted(colormap_rgb, alpha, orig_resized_rgb, 1 - alpha, 0)

            # Convert to PIL Image
            overlay_pil = Image.fromarray(overlay)

            # Convert to base64
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format="JPEG", quality=92)
            buffer.seek(0)
            b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"

        except Exception as e:
            logger.error("Error al generar Grad-CAM: %s", str(e), exc_info=True)
            return None
