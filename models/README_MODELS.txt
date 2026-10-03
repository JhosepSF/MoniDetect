========================================================================
                     MONIDETECT - MODEL DIRECTORY
========================================================================

Esta carpeta (`models/`) contiene los artefactos de modelos de aprendizaje
automático e inferencia fitosanitaria para MoniDetect:

1. cacao_yolo_segmenter.pt (o Segmentador_Cacao_YOLO26n_best.pt)
   - Modelo Ultralytics YOLO entrenado para segmentación de frutos de cacao.
   - Función: Detecta y genera la máscara del fruto para aislarlo sobre
     fondo blanco puro.

2. mobilenetv2_feature_extractor.keras (o mobilenetv2_segmented_final_extractor.keras)
   - Extractor de características MobileNetV2 afinado.
   - Función: Recibe la imagen segmentada RGB de 224x224 (preprocesada con
     preprocess_input) y produce un vector latente de 1280 características.

3. final_regressor.joblib / final_downstream_model.joblib (Modelo Ganador: SVR Linear)
   - Pipeline de regresión: StandardScaler + SVR(kernel='linear', C=1.0, epsilon=0.05).
   - Función: Estima la puntuación continua de severidad fitosanitaria:
       * Umbral de Decisión Óptimo: 0.43
       * Score < 0.43  => Sano (Clase 0)
       * Score >= 0.43 => Monilia (Clase 1)
   - Permite interpretabilidad directa y proyección en mapas de activación Grad-CAM.

4. model_metadata.json
   - Especificaciones y metadatos del pipeline de inferencia generado en el estudio
     (métricas de regresión, umbral óptimo 0.43, hiperparámetros).

5. mobilenetv2_segmented_final_finetuned.keras (Opcional)
   - Red convolucional completa MobileNetV2 ajustada para mapas de atención Grad-CAM.

------------------------------------------------------------------------
ESTRUCTURA FINAL ESPERADA:
------------------------------------------------------------------------
models/
├── README_MODELS.txt
├── cacao_yolo_segmenter.pt
├── mobilenetv2_feature_extractor.keras
├── final_regressor.joblib
├── final_downstream_model.joblib
└── model_metadata.json

NOTA: El sistema cuenta con resolución automática de alias y nombres de archivo
retrocompatibles.
========================================================================
