
"""
Preprocesamiento de imágenes para Banana_Guava_Classifier.

Compatible con MobileNetV3-Small y pesos preentrenados
de ImageNet.
"""

from pathlib import Path

from PIL import Image
from torchvision import transforms


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

TAMANO_IMAGEN = 224

# Estadísticas de ImageNet utilizadas por los pesos preentrenados.
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIACION_IMAGENET = [0.229, 0.224, 0.225]


# ==========================================================
# 2. TRANSFORMACIONES
# ==========================================================

def obtener_transformaciones():
    """
    Genera las transformaciones para entrenamiento,
    validación y prueba.

    El aumento de datos se aplica exclusivamente
    al conjunto de entrenamiento.
    """

    transformaciones = {

        "train": transforms.Compose([
            transforms.RandomResizedCrop(
                TAMANO_IMAGEN,
                scale=(0.80, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=MEDIA_IMAGENET,
                std=DESVIACION_IMAGENET
            )
        ]),

        "val": transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            transforms.CenterCrop(TAMANO_IMAGEN),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=MEDIA_IMAGENET,
                std=DESVIACION_IMAGENET
            )
        ]),

        "test": transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            transforms.CenterCrop(TAMANO_IMAGEN),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=MEDIA_IMAGENET,
                std=DESVIACION_IMAGENET
            )
        ])
    }

    return transformaciones


# ==========================================================
# 3. PREPARACIÓN DE UNA IMAGEN INDIVIDUAL
# ==========================================================

def preparar_imagen(ruta):
    """
    Prepara una imagen para realizar una predicción.

    Parámetro:
        ruta: ubicación de la imagen.

    Retorna:
        Tensor normalizado de tamaño [1, 3, 224, 224].
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {ruta}"
        )

    transformaciones = obtener_transformaciones()

    with Image.open(ruta) as imagen:
        imagen = imagen.convert("RGB")

        tensor = transformaciones["test"](imagen)

    # Agrega dimensión del lote (batch).
    return tensor.unsqueeze(0)
