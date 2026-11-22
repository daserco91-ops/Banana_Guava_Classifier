
"""
Entrenamiento V2 - Banana_Guava_Classifier

MobileNetV3-Small
Transfer learning + fine-tuning
Muestreo equilibrado de clases
Regularización y early stopping

No sobrescribe el modelo V1.
"""

from pathlib import Path
import random
import copy
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.models import MobileNet_V3_Small_Weights

from src.preprocessing import obtener_transformaciones


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

SEMILLA = 42

TAMANO_LOTE = 16

EPOCAS_CABEZA = 12
EPOCAS_FINE_TUNING = 15

PACIENCIA = 4

TASA_CABEZA = 0.0005
TASA_FINE_TUNING = 0.00005

FACTOR_REDUCCION_LR = 0.5
PACIENCIA_SCHEDULER = 2

PESO_DECAY = 0.0001
LABEL_SMOOTHING = 0.05

RAIZ = Path(__file__).resolve().parent.parent

RUTA_DATASET = RAIZ / "data" / "processed" / "dataset"
RUTA_MODELOS = RAIZ / "models"
RUTA_RESULTADOS = RAIZ / "results"

RUTA_CHECKPOINT = (
    RUTA_MODELOS / "mobilenetv3_banana_guava_v2.pth"
)

RUTA_HISTORIAL = (
    RUTA_RESULTADOS / "historial_entrenamiento_v2.csv"
)

RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device("cpu")

CLASES_ESPERADAS = ["Banana", "Guava"]


# ==========================================================
# 2. REPRODUCIBILIDAD
# ==========================================================

def fijar_semilla(semilla=SEMILLA):

    random.seed(semilla)
    np.random.seed(semilla)

    torch.manual_seed(semilla)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    clases = conjuntos["train"].classes

    if clases != CLASES_ESPERADAS:
        raise ValueError(
            f"Clases inesperadas: {clases}"
        )

    if conjuntos["val"].classes != clases:
        raise ValueError(
            "Las clases de train y val no coinciden."
        )

    # ------------------------------------------------------
    # Distribución de clases
    # ------------------------------------------------------

    etiquetas = conjuntos["train"].targets

    conteos = np.bincount(
        etiquetas,
        minlength=len(clases)
    )

    print("\nDISTRIBUCIÓN DEL ENTRENAMIENTO")

    for indice, clase in enumerate(clases):
        print(f"{clase}: {conteos[indice]} imágenes")

    # ------------------------------------------------------
    # Muestreo equilibrado
    # ------------------------------------------------------

    pesos_clase = 1.0 / torch.tensor(
        conteos,
        dtype=torch.float
    )

    pesos_muestras = pesos_clase[
        torch.tensor(etiquetas, dtype=torch.long)
    ]

    generador = torch.Generator()
    generador.manual_seed(SEMILLA)

    sampler = WeightedRandomSampler(
        weights=pesos_muestras,
        num_samples=len(pesos_muestras),
        replacement=True,
        generator=generador
    )

    cargadores = {

        "train": DataLoader(
            conjuntos["train"],
            batch_size=TAMANO_LOTE,
            sampler=sampler,
            num_workers=0
        ),

        "val": DataLoader(
            conjuntos["val"],
            batch_size=TAMANO_LOTE,
            shuffle=False,
            num_workers=0
        )
    }

    print("\nCONJUNTOS")

    for fase, conjunto in conjuntos.items():
        print(
            f"{fase}: {len(conjunto)} imágenes | "
            f"Clases: {conjunto.classes}"
        )

    return cargadores, clases


# ==========================================================
# 4. CREAR MODELO
# ==========================================================

def crear_modelo(numero_clases):

    pesos = MobileNet_V3_Small_Weights.DEFAULT

    modelo = models.mobilenet_v3_small(
        weights=pesos
    )

    # Congelar las capas convolucionales.
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False

    # Sustituir el clasificador final.
    entradas = modelo.classifier[-1].in_features

    modelo.classifier[-1] = nn.Linear(
        entradas,
        numero_clases
    )

    return modelo


# ==========================================================
# 5. CONTROL DE CAPAS ENTRENABLES
# ==========================================================

def configurar_fine_tuning(modelo):

    # Primero congelar todas las características.
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False

    # Descongelar los últimos tres bloques.
    for parametro in modelo.features[-3:].parameters():
        parametro.requires_grad = True

    # El clasificador siempre permanece entrenable.
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True

    entrenables = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    congelados = sum(
        p.numel()
        for p in modelo.parameters()
        if not p.requires_grad
    )

    print("\nPARÁMETROS DEL MODELO")
    print("Entrenables:", entrenables)
    print("Congelados:", congelados)


# ==========================================================
# 6. EJECUTAR UNA ÉPOCA
# ==========================================================

def ejecutar_epoca(
    modelo,
    cargador,
    criterio,
    optimizador=None
):

    entrenamiento = optimizador is not None

    if entrenamiento:
        modelo.train()
    else:
        modelo.eval()

    perdida_total = 0.0
    aciertos = 0
    total = 0

    contexto = (
        torch.enable_grad()
        if entrenamiento
        else torch.inference_mode()
    )

    with contexto:

        for imagenes, etiquetas in cargador:

            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)

            if entrenamiento:
                optimizador.zero_grad()

            salidas = modelo(imagenes)

            perdida = criterio(
                salidas,
                etiquetas
            )

            if entrenamiento:

                perdida.backward()

                optimizador.step()

            cantidad = etiquetas.size(0)

            perdida_total += perdida.item() * cantidad

            predicciones = salidas.argmax(dim=1)

            aciertos += (
                predicciones == etiquetas
            ).sum().item()

            total += cantidad

    if total == 0:
        raise RuntimeError("El cargador no contiene imágenes.")

    perdida_media = perdida_total / total

    exactitud = aciertos / total

    return perdida_media, exactitud


# ==========================================================
# 7. HISTORIAL
# ==========================================================

def guardar_historial(historial):

    campos = [
        "fase",
        "epoca",
        "perdida_train",
        "exactitud_train",
        "perdida_val",
        "exactitud_val",
        "tasa_aprendizaje"
    ]

    with open(
        RUTA_HISTORIAL,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(historial)


# ==========================================================
# 8. ENTRENAR UNA FASE
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
        lr=tasa_aprendizaje,
        weight_decay=PESO_DECAY
    )

    criterio = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="min",
        factor=FACTOR_REDUCCION_LR,
        patience=PACIENCIA_SCHEDULER
    )

    contador_sin_mejora = 0

    for epoca in range(1, epocas + 1):

        perdida_train, exactitud_train = ejecutar_epoca(
            modelo,
            cargadores["train"],
            criterio,
            optimizador
        )

        perdida_val, exactitud_val = ejecutar_epoca(
            modelo,
            cargadores["val"],
            criterio
        )

        scheduler.step(perdida_val)

        lr_actual = optimizador.param_groups[0]["lr"]

        registro = {
            "fase": fase,
            "epoca": epoca,
            "perdida_train": perdida_train,
            "exactitud_train": exactitud_train,
            "perdida_val": perdida_val,
            "exactitud_val": exactitud_val,
            "tasa_aprendizaje": lr_actual
        }

        historial.append(registro)

        guardar_historial(historial)

        print(
            f"\n{fase} | Época {epoca}/{epocas}"
        )

        print(
            f"Loss train: {perdida_train:.5f} | "
            f"Acc train: {exactitud_train:.2%}"
        )

        print(
            f"Loss val: {perdida_val:.5f} | "
            f"Acc val: {exactitud_val:.2%}"
        )

        print(
            f"Learning rate: {lr_actual:.7f}"
        )

        if perdida_val < mejor_perdida:

            mejor_perdida = perdida_val

            checkpoint = {
                "model_state_dict": copy.deepcopy(
                    modelo.state_dict()
                ),
                "clases": CLASES_ESPERADAS,
                "fase": fase,
                "epoca": epoca,
                "perdida_val": perdida_val,
                "exactitud_val": exactitud_val,
                "semilla": SEMILLA,
                "version": "V2"
            }

            torch.save(
                checkpoint,
                RUTA_CHECKPOINT
            )

            contador_sin_mejora = 0

            print("Mejor modelo V2 guardado.")

        else:

            contador_sin_mejora += 1

        if contador_sin_mejora >= PACIENCIA:

            print("Early stopping activado.")

            break

    return mejor_perdida


# ==========================================================
# 9. PROCESO PRINCIPAL
# ==========================================================

def main():

    fijar_semilla()

    print("=" * 60)
    print("BANANA_GUAVA_CLASSIFIER - ENTRENAMIENTO V2")
    print("=" * 60)

    print("Dispositivo:", DISPOSITIVO)

    cargadores, clases = cargar_datos()

    modelo = crear_modelo(len(clases))

    modelo = modelo.to(DISPOSITIVO)

    historial = []

    mejor_perdida = float("inf")

    # ------------------------------------------------------
    # FASE 1: CABEZA CLASIFICADORA
    # ------------------------------------------------------

    print("\nFASE 1: ENTRENAMIENTO DE LA CABEZA")

    mejor_perdida = entrenar_fase(
        modelo,
        cargadores,
        "cabeza",
        EPOCAS_CABEZA,
        TASA_CABEZA,
        historial,
        mejor_perdida
    )

    # Recuperar el mejor checkpoint de V2.
    checkpoint = torch.load(
        RUTA_CHECKPOINT,
        map_location=DISPOSITIVO,
        weights_only=True
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ------------------------------------------------------
    # FASE 2: FINE-TUNING
    # ------------------------------------------------------

    print("\nFASE 2: FINE-TUNING")

    configurar_fine_tuning(modelo)

    mejor_perdida = entrenar_fase(
        modelo,
        cargadores,
        "fine_tuning",
        EPOCAS_FINE_TUNING,
        TASA_FINE_TUNING,
        historial,
        mejor_perdida
    )

    # ------------------------------------------------------
    # FINAL
    # ------------------------------------------------------

    checkpoint_final = torch.load(
        RUTA_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    print("\nENTRENAMIENTO V2 FINALIZADO")
    print("=" * 60)

    print(
        "Mejor pérdida de validación:",
        checkpoint_final["perdida_val"]
    )

    print(
        "Exactitud de validación:",
        f"{checkpoint_final['exactitud_val']:.2%}"
    )

    print("Modelo:", RUTA_CHECKPOINT)
    print("Historial:", RUTA_HISTORIAL)


if __name__ == "__main__":
    main()
