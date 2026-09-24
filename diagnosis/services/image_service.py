"""
Image processing service for MoniDetect.
Handles image validation, YOLO cacao segmentation on white background,
botanical morphology & color validation, resizing (224x224 RGB),
MobileNetV2 preprocessing, and in-memory base64 encoding.
"""
import io
import base64
import logging
import numpy as np
from PIL import Image, ImageOps
from .model_service import CacaoNotDetectedError

logger = logging.getLogger(__name__)

# Allowed file extensions & MIME types
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ALLOWED_MIMETYPES = {'image/jpeg', 'image/png', 'image/pjpeg', 'image/x-png'}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class ImageService:
    """Service handling image operations for the MoniDetect pipeline."""

    @staticmethod
    def validate_image_file(uploaded_file):
        """
        Validates the uploaded file for:
        1. Max size limit (<= 10MB)
        2. Allowed extensions (JPG, JPEG, PNG)
        3. Real image file content & PIL integrity
        
        Raises ValueError with user-friendly Spanish message on validation failure.
        """
        if not uploaded_file:
            raise ValueError("No se ha seleccionado ningún archivo de imagen.")

        # Check size
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise ValueError(f"El tamaño del archivo supera el límite permitido ({max_mb} MB).")

        # Check filename extension
        filename = uploaded_file.name.lower()
        if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise ValueError("Formato no compatible. Solo se admiten archivos JPG, JPEG y PNG.")

        # Validate with Pillow
        try:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            img.verify()  # Verifies file header and stream integrity
            uploaded_file.seek(0)
        except Exception as e:
            logger.warning("Fallo en la validación de integridad de imagen: %s", str(e))
            raise ValueError("El archivo proporcionado no es una imagen válida o está dañado.") from e

    @staticmethod
    def file_to_pil(uploaded_file) -> Image.Image:
        """
        Reads an uploaded file into a PIL Image and corrects orientation using EXIF if needed.
        """
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        return image.convert('RGB')

    @staticmethod
    def validate_cacao_morphology_and_color(segmented_image: Image.Image):
        """
        Validates the segmented fruit against cacao botanical properties:
        1. Aspect ratio: Real cacao pods (Theobroma cacao) are elongated/fusiform/ellipsoid,
           with length-to-width aspect ratio >= 1.20. Round/spherical fruits (apples, oranges) are rejected.
        2. Color distribution: Cacao pods exhibit natural earthy hues (green, burgundy, brown, yellow-brown).
           Artificial neon or non-cacao color envelopes are filtered out.
        """
        import cv2

        seg_np = np.array(segmented_image)
        # Identify non-white pixels (the segmented fruit)
        mask = np.any(seg_np < 245, axis=2)
        y_idx, x_idx = np.where(mask)

        if len(y_idx) < 150:
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. Intente con otra fotografía."
            )

        box_h = y_idx.max() - y_idx.min() + 1
        box_w = x_idx.max() - x_idx.min() + 1
        aspect_ratio = max(box_h, box_w) / max(min(box_h, box_w), 1)

        # Aspect ratio filter: Cacao pods are oblong/elongated, not spherical
        if aspect_ratio < 1.20:
            logger.info("Objeto descartado por morfología no correspondiente a cacao (aspect_ratio=%.2f)", aspect_ratio)
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. "
                "La morfología redondeada no corresponde a una mazorca de cacao. Intente con otra fotografía."
            )

        # Color filter in HSV space
        fruit_pixels = seg_np[mask]
        hsv_pixels = cv2.cvtColor(fruit_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        h_mean = float(hsv_pixels[:, 0].mean())
        s_mean = float(hsv_pixels[:, 1].mean())

        # Filter out overly saturated commercial apple reds (H > 155, S > 185)
        if h_mean > 155 and s_mean > 185:
            logger.info("Objeto descartado por espectro de color no vegetal/cacao (H=%.1f, S=%.1f)", h_mean, s_mean)
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. "
                "Las tonalidades y textura no corresponden a una mazorca de cacao. Intente con otra fotografía."
            )

    @classmethod
    def segment_cacao_yolo(cls, image_pil: Image.Image, yolo_model, conf: float = 0.25) -> Image.Image:
        """
        Executes YOLO segmentation on the input image.
        - Locates and isolates the cacao fruit.
        - Combines multiple cacao masks if detected.
        - Replaces background outside mask with pure white (255, 255, 255).
        - Validates botanical morphology and color.
        - Returns a PIL Image with cacao on pure white background.
        """
        orig_img_rgb = np.array(image_pil.convert('RGB'))
        orig_h, orig_w, _ = orig_img_rgb.shape

        # Run YOLO inference
        logger.info("Ejecutando inferencia de segmentación YOLO con conf=%.2f", conf)
        results = yolo_model(orig_img_rgb, conf=conf, verbose=False)

        if not results or len(results) == 0:
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. Intente con otra fotografía."
            )

        result = results[0]

        # Verify that masks exist and at least one detection occurred
        if result.masks is None or len(result.masks) == 0:
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. Intente con otra fotografía."
            )

        # Extract mask tensors (shape: [N, H_mask, W_mask])
        masks_data = result.masks.data
        if masks_data is None or len(masks_data) == 0:
            raise CacaoNotDetectedError(
                "No se pudo identificar claramente un fruto de cacao en la imagen. Intente con otra fotografía."
            )

        try:
            import cv2
            # Combine all detected masks with logical OR
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            for i in range(len(masks_data)):
                mask_np = masks_data[i].cpu().numpy().astype(np.float32)
                # Resize mask to original image dimensions
                mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                # Threshold to binary
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                combined_mask = np.bitwise_or(combined_mask, binary_mask)

            # Check if mask is empty after resizing
            if np.sum(combined_mask) == 0:
                raise CacaoNotDetectedError(
                    "No se pudo identificar claramente un fruto de cacao en la imagen. Intente con otra fotografía."
                )

            # Create pure white background (255, 255, 255)
            white_background = np.full_like(orig_img_rgb, 255, dtype=np.uint8)

            # Composite: keep cacao inside mask, white outside
            mask_3d = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
            segmented_rgb = np.where(mask_3d == 1, orig_img_rgb, white_background)
            segmented_pil = Image.fromarray(segmented_rgb)

            # Botanical verification of cacao morphology and color
            cls.validate_cacao_morphology_and_color(segmented_pil)

            return segmented_pil

        except CacaoNotDetectedError:
            raise
        except Exception as e:
            logger.error("Error durante el post-procesamiento de máscara YOLO: %s", str(e), exc_info=True)
            raise ValueError(f"Error al procesar la segmentación del fruto: {str(e)}") from e

    @staticmethod
    def preprocess_for_mobilenet(segmented_image: Image.Image) -> np.ndarray:
        """
        Prepares the segmented white-background image for MobileNetV2:
        1. Converts to RGB (3 channels)
        2. Resizes to 224x224
        3. Converts to float32 numpy array
        4. Adds batch dimension -> (1, 224, 224, 3)
        5. Applies MobileNetV2 preprocess_input (-1 to 1 scaling)
        """
        # Ensure RGB
        img_rgb = segmented_image.convert('RGB')
        # Resize to 224x224
        img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to numpy float32
        img_array = np.array(img_resized, dtype=np.float32)

        # Add batch dimension -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # Apply tensorflow mobilenet_v2 preprocess_input
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        preprocessed_tensor = preprocess_input(img_batch)

        return preprocessed_tensor

    @staticmethod
    def image_to_base64(image: Image.Image, image_format: str = "JPEG", quality: int = 90) -> str:
        """
        Encodes a PIL Image to a Base64 data URI string for seamless in-memory transfer.
        """
        buffer = io.BytesIO()
        image.save(buffer, format=image_format, quality=quality)
        buffer.seek(0)
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime_type};base64,{b64_str}"
