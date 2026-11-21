# Banana_Guava_Classifier

Proyecto de clasificación de imágenes para distinguir **Banana** y **Guava** mediante visión por computadora y aprendizaje profundo.

## Objetivo inicial
Construir un pipeline reproducible que reciba una imagen, aplique el preprocesamiento, ejecute un modelo entrenado y devuelva la clase predicha: `Banana` o `Guava`.

## Datos
El conjunto de imágenes analizado previamente contiene 1,310 imágenes. Los nombres también incluyen etiquetas de calidad (`Class_A`, `Class_B`, `Defect`). La primera etapa de este proyecto se concentra en la clasificación binaria por fruta. La clasificación de calidad se considerará como una etapa independiente.

No se incluyen las imágenes en este repositorio. Coloca una copia del conjunto original en `data/raw/` y conserva intacta tu copia maestra.

## Estructura
- `data/raw/`: imágenes originales de trabajo (no se versionan).
- `notebooks/`: exploración y auditoría.
- `src/`: módulos reutilizables de Python.
- `models/`: modelos entrenados (no se versionan por defecto).
- `results/`: reportes y resultados.

## Inicio
1. Crea y activa un entorno virtual.
2. Instala dependencias con `pip install -r requirements.txt`.
3. Coloca las imágenes en `data/raw/`.
4. Abre y ejecuta `notebooks/02_Auditoria_Dataset.ipynb` desde la raíz del proyecto.

## Estado
Estructura inicial. El entrenamiento y las métricas deberán ejecutarse; no se consideran resultados hasta que se generen realmente.
