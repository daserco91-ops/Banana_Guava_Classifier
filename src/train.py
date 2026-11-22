
"""
Entrenamiento de Banana_Guava_Classifier
MobileNetV3-Small con transferencia de aprendizaje y fine-tuning.
"""

from pathlib import Path
import random
import copy
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import MobileNet_V3_Small_Weights

from src.preprocessing import obtener_transformaciones


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

SEMILLA = 42
TAMANO_LOTE = 16
EPOCAS_CABEZA = 10
EPOCAS_FINE_TUNING = 10
PACIENCIA = 3

TASA_CABEZA = 0.001
TASA_FINE_TUNING = 0.0001

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "processed" / "dataset"
RUTA_MODELOS = RAIZ / "models"
RUTA_RESULTADOS = RAIZ / "results"

RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device("cpu")


# ==========================================================
# 2. REPRODUCIBILIDAD
# ==========================================================

def fijar_semilla(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


# ==========================================================
# 3. CARGA DEL DATASET
# ==========================================================

def cargar_datos():

    transformaciones = obtener_transformaciones()

    conjuntos = {
        fase: datasets.ImageFolder(
            RUTA_DATASET / fase,
            transform=transformaciones[fase]
        )
        for fase in ["train", "val"]
    }

    for fase, conjunto in conjuntos.items():
        print(
            f"{fase}: {len(conjunto)} imágenes | "
            f"Clases: {conjunto.classes}"
        )

    clases = conjuntos["train"].classes

    if clases != ["Banana", "Guava"]:
        raise ValueError(
            f"Clases inesperadas: {clases}. "
            "Se esperaban Banana y Guava."
        )

    if conjuntos["val"].classes != clases:
        raise ValueError("Las clases de train y val no coinciden.")

    generador = torch.Generator().manual_seed(SEMILLA)

    cargadores = {
        "train": DataLoader(
            conjuntos["train"],
            batch_size=TAMANO_LOTE,
            shuffle=True,
            num_workers=0,
            generator=generador
        ),
        "val": DataLoader(
            conjuntos["val"],
            batch_size=TAMANO_LOTE,
            shuffle=False,
            num_workers=0
        )
    }

    return cargadores, clases


# ==========================================================
# 4. MODELO MOBILENETV3-SMALL
# ==========================================================

def crear_modelo(numero_clases):

    pesos = MobileNet_V3_Small_Weights.DEFAULT

    modelo = models.mobilenet_v3_small(weights=pesos)

    # Congelar todas las capas de extracción.
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False

    # Sustituir la última capa por nuestro clasificador.
    entradas = modelo.classifier[-1].in_features

    modelo.classifier[-1] = nn.Linear(
        entradas,
        numero_clases
    )

    return modelo


# ==========================================================
# 5. ENTRENAMIENTO Y VALIDACIÓN
# ==========================================================

def ejecutar_epoca(modelo, cargador, criterio, optimizador=None):

    entrenamiento = optimizador is not None

    modelo.train() if entrenamiento else modelo.eval()

    perdida_total = 0.0
    aciertos = 0
    total = 0

    contexto = torch.enable_grad() if entrenamiento else torch.no_grad()

    with contexto:

        for imagenes, etiquetas in cargador:

            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)

            if entrenamiento:
                optimizador.zero_grad()

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            if entrenamiento:
                perdida.backward()
                optimizador.step()

            cantidad = etiquetas.size(0)

            perdida_total += perdida.item() * cantidad

            predicciones = salidas.argmax(dim=1)

            aciertos += (predicciones == etiquetas).sum().item()
            total += cantidad

    perdida_media = perdida_total / total
    precision = aciertos / total

    return perdida_media, precision


# ==========================================================
# 6. GUARDAR HISTORIAL
# ==========================================================

def guardar_historial(historial):

    ruta = RUTA_RESULTADOS / "historial_entrenamiento.csv"

    campos = [
        "fase",
        "epoca",
        "perdida_train",
        "precision_train",
        "perdida_val",
        "precision_val"
    ]

    with open(ruta, "w", newline="", encoding="utf-8") as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(historial)


# ==========================================================
# 7. ENTRENAMIENTO POR FASES
# ==========================================================

def entrenar_fase(
    modelo,
    cargadores,
    fase,
    epocas,
    tasa_aprendizaje,
    historial,
    mejor_perdida
):

    parametros = [
        p for p in modelo.parameters()
        if p.requires_grad
    ]

    optimizador = torch.optim.AdamW(
        parametros,
        lr=tasa_aprendizaje
    )

    criterio = nn.CrossEntropyLoss()

    mejor_estado = None
    contador_sin_mejora = 0

    for epoca in range(1, epocas + 1):

        perdida_train, precision_train = ejecutar_epoca(
            modelo,
            cargadores["train"],
            criterio,
            optimizador
        )

        perdida_val, precision_val = ejecutar_epoca(
            modelo,
            cargadores["val"],
            criterio
        )

        registro = {
            "fase": fase,
            "epoca": epoca,
            "perdida_train": perdida_train,
            "precision_train": precision_train,
            "perdida_val": perdida_val,
            "precision_val": precision_val
        }

        historial.append(registro)
        guardar_historial(historial)

        print(
            f"{fase} | Época {epoca}/{epocas} | "
            f"Loss train: {perdida_train:.4f} | "
            f"Acc train: {precision_train:.4f} | "
            f"Loss val: {perdida_val:.4f} | "
            f"Acc val: {precision_val:.4f}"
        )

        if perdida_val < mejor_perdida:

            mejor_perdida = perdida_val
            mejor_estado = copy.deepcopy(modelo.state_dict())

            checkpoint = {
                "model_state_dict": mejor_estado,
                "clases": ["Banana", "Guava"],
                "fase": fase,
                "epoca": epoca,
                "perdida_val": perdida_val
            }

            torch.save(
                checkpoint,
                RUTA_MODELOS / "mobilenetv3_banana_guava.pth"
            )

            contador_sin_mejora = 0

            print("  Mejor modelo guardado.")

        else:

            contador_sin_mejora += 1

        if contador_sin_mejora >= PACIENCIA:
            print("  Early stopping activado.")
            break

    return mejor_perdida, mejor_estado


# ==========================================================
# 8. PROCESO PRINCIPAL
# ==========================================================

def main():

    fijar_semilla()

    print("=" * 55)
    print("BANANA_GUAVA_CLASSIFIER")
    print("Entrenamiento MobileNetV3-Small")
    print("=" * 55)

    print("Dispositivo:", DISPOSITIVO)

    cargadores, clases = cargar_datos()

    modelo = crear_modelo(len(clases))
    modelo = modelo.to(DISPOSITIVO)

    historial = []
    mejor_perdida = float("inf")

    # FASE 1: entrenar solamente el clasificador.
    print("\nFASE 1: CABEZA CLASIFICADORA")

    mejor_perdida, _ = entrenar_fase(
        modelo,
        cargadores,
        "cabeza",
        EPOCAS_CABEZA,
        TASA_CABEZA,
        historial,
        mejor_perdida
    )

    # Recuperar el mejor modelo antes de continuar.
    checkpoint = torch.load(
        RUTA_MODELOS / "mobilenetv3_banana_guava.pth",
        map_location=DISPOSITIVO,
        weights_only=True
    )

    modelo.load_state_dict(checkpoint["model_state_dict"])

    # FASE 2: descongelar las últimas capas convolucionales.
    print("\nFASE 2: FINE-TUNING")

    for parametro in modelo.features[-3:].parameters():
        parametro.requires_grad = True

    mejor_perdida, _ = entrenar_fase(
        modelo,
        cargadores,
        "fine_tuning",
        EPOCAS_FINE_TUNING,
        TASA_FINE_TUNING,
        historial,
        mejor_perdida
    )

    print("\nENTRENAMIENTO FINALIZADO")
    print("Mejor pérdida de validación:", mejor_perdida)
    print("Modelo:", RUTA_MODELOS / "mobilenetv3_banana_guava.pth")
    print("Historial:", RUTA_RESULTADOS / "historial_entrenamiento.csv")


if __name__ == "__main__":
    main()
