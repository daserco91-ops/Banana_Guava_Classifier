"""Funciones de preprocesamiento para Banana_Guava_Classifier."""

from PIL import Image


def cargar_imagen(ruta):
    """Abre una imagen y la convierte a RGB."""
    with Image.open(ruta) as imagen:
        return imagen.convert("RGB")


def preparar_imagen(ruta, tamano=(224, 224)):
    """Carga y redimensiona una imagen; devuelve una imagen PIL RGB."""
    imagen = cargar_imagen(ruta)
    return imagen.resize(tamano)
