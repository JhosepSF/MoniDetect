import io
import json
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from .services.model_service import ModelManager, ModelNotFoundError
from .services.image_service import ImageService
from .services.inference_service import InferenceService

class MoniDetectUnitTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.manager = ModelManager()

    def test_model_manager_missing_models_detection(self):
        """Verifies that ModelManager correctly flags missing models without crashing."""
        status = self.manager.check_models_status()
        self.assertIn('Segmentador_Cacao_YOLO26n_best.pt', status['missing_required'])
        self.assertIn('mobilenetv2_segmented_final_extractor.keras', status['missing_required'])
        self.assertIn('segmented_svc_final.joblib', status['missing_required'])
        self.assertIn('mobilenetv2_segmented_final_finetuned.keras', status['missing_optional'])
        self.assertFalse(status['is_ready_for_inference'])

    def test_image_validation_rejects_invalid_extension(self):
        """Verifies that invalid extensions like .txt are rejected."""
        dummy_file = SimpleUploadedFile("test.txt", b"not an image", content_type="text/plain")
        with self.assertRaises(ValueError) as ctx:
            ImageService.validate_image_file(dummy_file)
        self.assertIn("Formato no compatible", str(ctx.exception))

    def test_image_validation_accepts_valid_image(self):
        """Verifies that a valid JPEG image passes validation."""
        img = Image.new('RGB', (100, 100), color=(120, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        uploaded = SimpleUploadedFile("sample.jpg", buf.getvalue(), content_type="image/jpeg")
        # Should not raise any exception
        ImageService.validate_image_file(uploaded)

    def test_inference_service_raises_controlled_error_on_missing_models(self):
        """Verifies that InferenceService gracefully raises ModelNotFoundError when models are missing."""
        img = Image.new('RGB', (100, 100), color=(120, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        uploaded = SimpleUploadedFile("sample.jpg", buf.getvalue(), content_type="image/jpeg")

        with self.assertRaises(ModelNotFoundError) as ctx:
            InferenceService.predict_cacao(uploaded)
        self.assertIn("Faltan los siguientes modelos", str(ctx.exception))

    def test_index_view_renders_successfully(self):
        """Verifies that the index view renders with HTTP 200 and includes MoniDetect branding."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'MoniDetect')
        self.assertContains(response, 'Detección inteligente de moniliasis en frutos de cacao')
        self.assertContains(response, 'Modelos Pendientes')

    def test_api_status_endpoint(self):
        """Verifies the /api/status/ JSON endpoint."""
        response = self.client.get('/api/status/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertFalse(data['is_ready_for_inference'])
        self.assertEqual(len(data['missing_required']), 3)

    def test_api_analyze_endpoint_with_missing_models(self):
        """Verifies the /api/analyze/ endpoint returns 503 when required models are missing."""
        img = Image.new('RGB', (100, 100), color=(120, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        uploaded = SimpleUploadedFile("sample.jpg", buf.getvalue(), content_type="image/jpeg")

        response = self.client.post('/api/analyze/', {'image': uploaded})
        self.assertEqual(response.status_code, 503)
        data = json.loads(response.content)
        self.assertFalse(data['success'])
        self.assertEqual(data['type'], 'model_not_found')
