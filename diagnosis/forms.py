"""
Forms for the MoniDetect diagnosis app.
"""
from django import forms
from .services.image_service import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES

class ImageUploadForm(forms.Form):
    """Form to validate and handle cacao image uploads."""
    image = forms.FileField(
        required=True,
        label="Fotografía del fruto de cacao",
        widget=forms.FileInput(attrs={
            'accept': '.jpg,.jpeg,.png',
            'id': 'image-file-input',
            'class': 'file-input-hidden'
        })
    )
    include_gradcam = forms.BooleanField(
        required=False,
        initial=False,
        label="Mostrar mapa de atención (Grad-CAM)",
        widget=forms.CheckboxInput(attrs={
            'id': 'gradcam-toggle',
            'class': 'toggle-checkbox'
        })
    )

    def clean_image(self):
        file = self.cleaned_data.get('image')
        if not file:
            raise forms.ValidationError("Debe seleccionar una imagen.")

        if file.size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise forms.ValidationError(f"El archivo excede el tamaño máximo permitido de {max_mb} MB.")

        filename = file.name.lower()
        if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise forms.ValidationError("Formato inválido. Solo se admiten archivos en formato JPG, JPEG o PNG.")

        return file
