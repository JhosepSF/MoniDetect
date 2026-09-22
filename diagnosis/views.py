"""
Views for the MoniDetect cacao moniliasis diagnosis web application.
"""
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .forms import ImageUploadForm
from .services.model_service import ModelManager, ModelNotFoundError, ModelLoadError, CacaoNotDetectedError
from .services.inference_service import InferenceService

logger = logging.getLogger(__name__)

def index_view(request):
    """
    Renders the main MoniDetect landing and analysis dashboard.
    Checks model availability status and injects it into context.
    """
    manager = ModelManager()
    model_status = manager.check_models_status()
    form = ImageUploadForm()

    context = {
        'form': form,
        'model_status': model_status,
        'is_ready': model_status['is_ready_for_inference'],
        'missing_required': model_status['missing_required'],
        'missing_optional': model_status['missing_optional'],
    }
    return render(request, 'diagnosis/index.html', context)


@require_http_methods(["POST"])
def analyze_view(request):
    """
    API endpoint for running inference on an uploaded cacao image.
    Supports asynchronous AJAX JSON responses and standard form handling.
    """
    form = ImageUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        errors = [err for error_list in form.errors.values() for err in error_list]
        error_msg = " ".join(errors) if errors else "Datos de formulario inválidos."
        return JsonResponse({
            'success': False,
            'error': error_msg,
            'type': 'form_validation_error'
        }, status=400)

    image_file = form.cleaned_data['image']
    include_gradcam = form.cleaned_data.get('include_gradcam', False)

    try:
        result = InferenceService.predict_cacao(image_file, include_gradcam=include_gradcam)
        return JsonResponse(result, status=200)

    except ModelNotFoundError as e:
        logger.warning("Intento de inferencia sin modelos requeridos: %s", str(e))
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': 'model_not_found',
            'missing_models': getattr(e, 'missing_models', [])
        }, status=503)

    except CacaoNotDetectedError as e:
        logger.info("YOLO no detectó fruto de cacao: %s", str(e))
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': 'cacao_not_found'
        }, status=400)

    except ValueError as e:
        logger.warning("Error de validación de imagen: %s", str(e))
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': 'image_validation_error'
        }, status=400)

    except ModelLoadError as e:
        logger.error("Error al cargar modelo en memoria: %s", str(e), exc_info=True)
        return JsonResponse({
            'success': False,
            'error': f"Error técnico al inicializar los modelos: {str(e)}",
            'type': 'model_load_error'
        }, status=500)

    except Exception as e:
        logger.error("Error no controlado durante la inferencia: %s", str(e), exc_info=True)
        return JsonResponse({
            'success': False,
            'error': "Ocurrió un error inesperado durante el procesamiento de la imagen. Por favor, intente con otra fotografía.",
            'type': 'server_error'
        }, status=500)


@require_http_methods(["GET"])
def model_status_view(request):
    """
    Returns the real-time availability status of all model files in JSON format.
    """
    manager = ModelManager()
    status = manager.check_models_status()
    return JsonResponse(status, status=200)
