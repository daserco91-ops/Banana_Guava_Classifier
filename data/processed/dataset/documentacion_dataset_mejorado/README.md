# Dataset mejorado: Banana vs. Guava

## Objetivo
Clasificación binaria de imágenes: Banana o Guava.

## Estructura
- `dataset/train/Banana` y `dataset/train/Guava`
- `dataset/val/Banana` y `dataset/val/Guava`
- `dataset/test/Banana` y `dataset/test/Guava`

## Depuración
- 1,310 imágenes originales examinadas.
- 1,310 imágenes legibles; JPEG, RGB, 1024 x 1024.
- 38 pares de duplicados exactos detectados con SHA-256 (76 archivos involucrados).
- Se conservó un representante por hash: 1,272 imágenes únicas.
- Los 38 pares duplicados son de Banana; no hay duplicados entre frutas.
- Los archivos originales no se modificaron.
- Las etiquetas de calidad originales se conservaron como metadatos, pero el objetivo actual es solo Banana vs. Guava.

## División estratificada reproducible (semilla 42)
- Train: 890 (Banana 662, Guava 228)
- Val: 191 (Banana 142, Guava 49)
- Test: 191 (Banana 142, Guava 49)

La división se hizo después de agrupar por SHA-256. Ningún duplicado exacto se reparte entre particiones.

## Archivos de trazabilidad
- `auditoria_original_1310.csv`: una fila por archivo original.
- `manifiesto_dataset_limpio_1272.csv`: una fila por imagen única y su partición.
- `duplicados_exactos_38_pares.csv`: detalle de las 76 copias duplicadas.
- `resumen_division.csv`: distribución por clase y partición.

## Advertencia
En los grupos duplicados de Banana aparecen etiquetas de calidad diferentes (Class_A, Class_B y Defect). Para el clasificador binario no hay conflicto de fruta, pero las etiquetas de calidad deben revisarse manualmente antes de entrenar un modelo de calidad.
