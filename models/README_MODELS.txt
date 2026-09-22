========================================================================
                     MONIDETECT - MODEL DIRECTORY
========================================================================

Por favor, copia en esta carpeta (`models/`) los siguientes cuatro (4)
archivos de modelos entrenados para que el sistema de inferencia funcione:

1. Segmentador_Cacao_YOLO26n_best.pt
   - Modelo Ultralytics YOLO entrenado para segmentación de frutos de cacao.
   - Función: Detecta y genera la máscara del fruto para aislarlo sobre
     fondo blanco.

2. mobilenetv2_segmented_final_extractor.keras
   - Extractor de características MobileNetV2 ajustado.
   - Función: Recibe la imagen segmentada RGB de 224x224 (preprocesada con
     preprocess_input) y produce un vector de 1280 características.

3. segmented_svc_final.joblib
   - Pipeline de clasificación que integra StandardScaler + SVC.
   - Función: Clasifica el vector de 1280 características en:
       * 0 = Sano (Healthy)
       * 1 = Monilia

4. mobilenetv2_segmented_final_finetuned.keras
   - Red neuronal convolucional completa MobileNetV2 afinada.
   - Función: Utilizada para la generación opcional de mapas de calor
     Grad-CAM (activación de regiones de la CNN).

------------------------------------------------------------------------
ESTRUCTURA FINAL ESPERADA:
------------------------------------------------------------------------
models/
├── README_MODELS.txt
├── Segmentador_Cacao_YOLO26n_best.pt
├── mobilenetv2_segmented_final_extractor.keras
├── mobilenetv2_segmented_final_finetuned.keras
└── segmented_svc_final.joblib

NOTA: Si falta alguno de los 3 modelos principales (YOLO, Extractor, SVC),
el servidor MoniDetect no se caerá, pero notificará claramente al usuario
qué archivo debe ser colocado en esta carpeta.
========================================================================
