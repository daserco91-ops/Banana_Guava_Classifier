
"""Pipeline de inferencia para Banana_Guava_Classifier."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.preprocessing import preparar_imagen


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

RUTA_MODELO = Path("models/mobilenetv3_banana_guava.pth")
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 2. CARGA DEL MODELO
# ==========================================================

def cargar_modelo():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No existe el modelo: {RUTA_MODELO}")

    checkpoint = torch.load(
        RUTA_MODELO,
        map_location=DISPOSITIVO,
        weights_only=True
    )

    clases = checkpoint["clases"]

    modelo = models.mobilenet_v3_small(weights=None)
    entradas = modelo.classifier[-1].in_features
    modelo.classifier[-1] = nn.Linear(entradas, len(clases))

    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(DISPOSITIVO)
    modelo.eval()

    return modelo, clases


# ==========================================================
# 3. PREDICCIÓN DE UNA IMAGEN
# ==========================================================

def predecir_imagen(ruta_imagen):
    modelo, clases = cargar_modelo()

    tensor = preparar_imagen(ruta_imagen).to(DISPOSITIVO)

    with torch.inference_mode():
        salidas = modelo(tensor)
        probabilidades = torch.softmax(salidas, dim=1)[0]

    indice = probabilidades.argmax().item()

    return {
        "imagen": str(ruta_imagen),
        "prediccion": clases[indice],
        "confianza": float(probabilidades[indice].item()),
        "probabilidades": {
            clase: float(probabilidades[i].item())
            for i, clase in enumerate(clases)
        }
    }


# ==========================================================
# 4. EJECUCIÓN
# ==========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clasificador de bananas y guayabas"
    )
    parser.add_argument("imagen", help="Ruta de la imagen que se clasificará")

    argumentos = parser.parse_args()

    resultado = predecir_imagen(argumentos.imagen)

    print("\nRESULTADO DE LA CLASIFICACIÓN")
    print("--------------------------------")
    print("Imagen:", resultado["imagen"])
    print("Predicción:", resultado["prediccion"])
    print(f"Confianza: {resultado['confianza']:.2%}")

    print("\nProbabilidades:")
    for clase, probabilidad in resultado["probabilidades"].items():
        print(f"{clase}: {probabilidad:.2%}")
