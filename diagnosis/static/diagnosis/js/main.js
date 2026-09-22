/**
 * MoniDetect - Frontend Interaction & Inference Controller
 * Handles drag-and-drop, image preview, AJAX inference requests, and dynamic DOM updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Upload & Dropzone
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('image-file-input');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const dropzonePreview = document.getElementById('dropzone-preview');
    const previewImg = document.getElementById('preview-img-element');
    const previewFilename = document.getElementById('preview-filename');
    const previewFilesize = document.getElementById('preview-filesize');
    const btnClearPreview = document.getElementById('btn-clear-preview');
    
    // DOM Elements - Controls & Options
    const diagnosisForm = document.getElementById('diagnosis-form');
    const gradcamToggle = document.getElementById('include_gradcam');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnReset = document.getElementById('btn-reset');
    const statusMsgContainer = document.getElementById('status-message-container');
    const statusMsg = document.getElementById('status-message');

    // DOM Elements - Results & States
    const resultsEmptyView = document.getElementById('results-empty-view');
    const resultsLoadingView = document.getElementById('results-loading-view');
    const resultsContentView = document.getElementById('results-content-view');
    const loadingStepText = document.getElementById('loading-step-text');
    const pipelineStatusBadge = document.getElementById('pipeline-status-badge');

    // DOM Elements - Outcome & Visuals
    const outcomeBox = document.getElementById('diagnosis-outcome-box');
    const outcomeBadge = document.getElementById('outcome-badge');
    const outcomeMessageText = document.getElementById('outcome-message-text');
    const metricDecisionScore = document.getElementById('metric-decision-score');
    const resultOriginalImg = document.getElementById('result-original-img');
    const resultSegmentedImg = document.getElementById('result-segmented-img');
    const resultGradcamImg = document.getElementById('result-gradcam-img');
    const galleryGradcamItem = document.getElementById('gallery-gradcam-item');
    const disclaimerText = document.getElementById('disclaimer-text');

    // Allowed file types & Max size (10 MB)
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
    const MAX_SIZE_BYTES = 10 * 1024 * 1024;

    let currentFile = null;

    // =========================================================================
    // Dropzone & File Selection Handlers
    // =========================================================================

    // Click on dropzone triggers file input
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#btn-clear-preview')) return;
        fileInput.click();
    });

    // Keyboard accessibility for dropzone
    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Drag over / enter
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        }, false);
    });

    // Drag leave / drop
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleSelectedFile(fileInput.files[0]);
        }
    });

    // Clear preview button
    btnClearPreview.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadState();
    });

    // Reset button
    btnReset.addEventListener('click', () => {
        resetFullState();
    });

    // =========================================================================
    // File Validation & Preview Logic
    // =========================================================================

    function handleSelectedFile(file) {
        hideStatusMessage();

        // 1. Validate MIME type or extension
        const isValidType = ALLOWED_TYPES.includes(file.type) || 
                            /\.(jpg|jpeg|png)$/i.test(file.name);
        if (!isValidType) {
            showStatusMessage('Formato no compatible. Por favor seleccione una imagen en formato JPG, JPEG o PNG.', 'error');
            return;
        }

        // 2. Validate file size
        if (file.size > MAX_SIZE_BYTES) {
            showStatusMessage('El archivo excede el tamaño máximo permitido de 10 MB.', 'error');
            return;
        }

        currentFile = file;

        // Render preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = formatBytes(file.size);

            dropzonePrompt.classList.add('hidden');
            dropzonePreview.classList.remove('hidden');

            btnAnalyze.disabled = false;
            btnReset.disabled = false;
            pipelineStatusBadge.textContent = 'Imagen lista para análisis';
        };
        reader.readAsDataURL(file);
    }

    function resetUploadState() {
        currentFile = null;
        fileInput.value = '';
        previewImg.src = '';
        dropzonePreview.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        btnAnalyze.disabled = true;
    }

    function resetFullState() {
        resetUploadState();
        btnReset.disabled = true;
        hideStatusMessage();
        showResultsState('empty');
        pipelineStatusBadge.textContent = 'Esperando imagen';
    }

    // =========================================================================
    // Inference Form Submission (AJAX)
    // =========================================================================

    diagnosisForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!currentFile) {
            showStatusMessage('Por favor seleccione una imagen antes de analizar.', 'error');
            return;
        }

        hideStatusMessage();
        showResultsState('loading');
        btnAnalyze.disabled = true;
        btnReset.disabled = true;

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('include_gradcam', gradcamToggle.checked ? 'true' : 'false');
        formData.append('csrfmiddlewaretoken', window.CSRF_TOKEN);

        try {
            const response = await fetch(window.ANALYZE_URL, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const data = await response.json();

            if (response.ok && data.success) {
                renderResults(data);
                showResultsState('content');
                pipelineStatusBadge.textContent = 'Diagnóstico completado';
            } else {
                showResultsState('empty');
                const errorMessage = data.error || 'Ocurrió un problema al procesar la solicitud.';
                showStatusMessage(errorMessage, 'error');
                pipelineStatusBadge.textContent = 'Error en el análisis';
            }

        } catch (err) {
            console.error('Error de comunicación con el servidor:', err);
            showResultsState('empty');
            showStatusMessage('No se pudo establecer conexión con el servidor. Verifique que el servicio esté activo.', 'error');
            pipelineStatusBadge.textContent = 'Error de conexión';
        } finally {
            btnAnalyze.disabled = false;
            btnReset.disabled = false;
        }
    });

    // =========================================================================
    // UI Renderers & State Switchers
    // =========================================================================

    function showResultsState(state) {
        resultsEmptyView.classList.add('hidden');
        resultsLoadingView.classList.add('hidden');
        resultsContentView.classList.add('hidden');

        if (state === 'empty') {
            resultsEmptyView.classList.remove('hidden');
        } else if (state === 'loading') {
            resultsLoadingView.classList.remove('hidden');
        } else if (state === 'content') {
            resultsContentView.classList.remove('hidden');
        }
    }

    function renderResults(data) {
        // Diagnosis Outcome Box
        outcomeBox.className = 'diagnosis-outcome-box';
        const isMonilia = (data.class_id === 1 || data.class_name.toLowerCase() === 'monilia');

        if (isMonilia) {
            outcomeBox.classList.add('outcome-monilia');
            outcomeBadge.textContent = 'MONILIA';
        } else {
            outcomeBox.classList.add('outcome-sano');
            outcomeBadge.textContent = 'SANO';
        }

        outcomeMessageText.textContent = data.message || 
            (isMonilia 
                ? 'Se detectaron características compatibles con moniliasis en el fruto analizado.' 
                : 'No se detectaron características compatibles con moniliasis en el fruto analizado.');

        // Decision Score (SVC)
        const scoreVal = data.decision_score_display || Number(data.decision_score).toFixed(4);
        metricDecisionScore.textContent = scoreVal;

        // Original Image
        resultOriginalImg.src = data.original_image;

        // Segmented Image (YOLO White Background)
        resultSegmentedImg.src = data.segmented_image;

        // Grad-CAM Attention Map (if available)
        if (data.has_gradcam && data.gradcam_image) {
            resultGradcamImg.src = data.gradcam_image;
            galleryGradcamItem.classList.remove('hidden');
        } else {
            galleryGradcamItem.classList.add('hidden');
            resultGradcamImg.src = '';
        }

        // Disclaimer
        if (data.disclaimer) {
            disclaimerText.textContent = data.disclaimer;
        }
    }

    function showStatusMessage(text, type = 'error') {
        statusMsg.className = `alert-banner banner-${type}`;
        statusMsg.textContent = text;
        statusMsgContainer.classList.remove('hidden');
        statusMsgContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideStatusMessage() {
        statusMsgContainer.classList.add('hidden');
        statusMsg.textContent = '';
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
});
