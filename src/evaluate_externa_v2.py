
"""
Evaluación externa V2 - Banana_Guava_Classifier

Evalúa exclusivamente el checkpoint V2.
No modifica el entrenamiento ni el modelo V1.
"""

from pathlib import Path
import csv

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.preprocessing import preparar_imagen


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

RAIZ = Path(__file__).resolve().parent.parent

RUTA_DATASET = RAIZ / "data" / "external_test"

RUTA_MODELO = (
    RAIZ / "models" /
    "mobilenetv3_banana_guava_v2.pth"
)

RUTA_RESULTADOS = RAIZ / "results"

RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

RUTA_CSV = RUTA_RESULTADOS / "evaluacion_externa_v2.csv"

RUTA_METRICAS = RUTA_RESULTADOS / "metricas_externas_v2.txt"

RUTA_MATRIZ = RUTA_RESULTADOS / "matriz_confusion_externa_v2.png"

CLASES_VALIDAS = ["Banana", "Guava"]

EXTENSIONES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}

DISPOSITIVO = torch.device("cpu")


# ==========================================================
# 2. CARGAR MODELO V2
# ==========================================================

def cargar_modelo_v2():

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    checkpoint = torch.load(
        RUTA_MODELO,
        map_location=DISPOSITIVO,
        weights_only=True
    )

    if checkpoint.get("version") != "V2":
        raise ValueError("El checkpoint no corresponde a V2.")

    clases = checkpoint["clases"]

    if clases != CLASES_VALIDAS:
        raise ValueError(f"Orden de clases incorrecto: {clases}")

    modelo = models.mobilenet_v3_small(weights=None)

    entradas = modelo.classifier[-1].in_features

    modelo.classifier[-1] = nn.Linear(
        entradas,
        len(clases)
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo.to(DISPOSITIVO)
    modelo.eval()

    print("Modelo V2 cargado correctamente.")
    print("Clases:", clases)

    return modelo, clases


# ==========================================================
# 3. OBTENER IMÁGENES EXTERNAS
# ==========================================================

def obtener_imagenes():

    imagenes = []

    for carpeta in sorted(RUTA_DATASET.iterdir()):

        if not carpeta.is_dir():
            continue

        nombre = carpeta.name.lower()

        if nombre == "banana":
            etiqueta = "Banana"

        elif nombre == "guava":
            etiqueta = "Guava"

        else:
            continue

        for ruta in carpeta.rglob("*"):

            if (
                ruta.is_file()
                and ruta.suffix.lower() in EXTENSIONES
            ):

                imagenes.append((ruta, etiqueta))

    return sorted(
        imagenes,
        key=lambda elemento: str(elemento[0])
    )


# ==========================================================
# 4. EVALUACIÓN
# ==========================================================

def evaluar():

    modelo, clases = cargar_modelo_v2()

    imagenes = obtener_imagenes()

    if len(imagenes) != 27:
        raise ValueError(
            f"Se esperaban 27 imágenes únicas, "
            f"pero se encontraron {len(imagenes)}."
        )

    etiquetas_reales = []
    predicciones = []
    registros = []

    print("\nEVALUACIÓN EXTERNA V2")
    print("=" * 55)

    print("Imágenes:", len(imagenes))
    print("Dispositivo:", DISPOSITIVO)

    for numero, (ruta, etiqueta_real) in enumerate(
        imagenes, start=1
    ):

        tensor = preparar_imagen(ruta).to(DISPOSITIVO)

        with torch.inference_mode():

            salidas = modelo(tensor)

            probabilidades = torch.softmax(
                salidas,
                dim=1
            )[0]

        indice = probabilidades.argmax().item()

        prediccion = clases[indice]

        confianza = probabilidades[indice].item()

        correcta = etiqueta_real == prediccion

        etiquetas_reales.append(etiqueta_real)
        predicciones.append(prediccion)

        registros.append({
            "imagen": str(ruta.relative_to(RAIZ)),
            "etiqueta_real": etiqueta_real,
            "prediccion": prediccion,
            "confianza": round(confianza, 6),
            "correcta": correcta
        })

        print(
            f"[{numero:02d}/27] "
            f"Real: {etiqueta_real:7} | "
            f"Predicción: {prediccion:7} | "
            f"Confianza: {confianza:.2%}"
        )

    # ======================================================
    # 5. MÉTRICAS
    # ======================================================

    exactitud = accuracy_score(
        etiquetas_reales,
        predicciones
    )

    matriz = confusion_matrix(
        etiquetas_reales,
        predicciones,
        labels=CLASES_VALIDAS
    )

    reporte = classification_report(
        etiquetas_reales,
        predicciones,
        labels=CLASES_VALIDAS,
        target_names=CLASES_VALIDAS,
        zero_division=0
    )

    aciertos = sum(
        registro["correcta"]
        for registro in registros
    )

    errores = len(registros) - aciertos

    print("\nRESUMEN V2")
    print("=" * 55)

    print("Total:", len(registros))
    print("Aciertos:", aciertos)
    print("Errores:", errores)
    print(f"Exactitud: {exactitud:.2%}")

    print("\nREPORTE DE CLASIFICACIÓN")
    print(reporte)

    # ======================================================
    # 6. EXPORTAR CSV
    # ======================================================

    with open(
        RUTA_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        campos = [
            "imagen",
            "etiqueta_real",
            "prediccion",
            "confianza",
            "correcta"
        ]

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(registros)

    # ======================================================
    # 7. EXPORTAR MÉTRICAS
    # ======================================================

    with open(
        RUTA_METRICAS,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write("EVALUACIÓN EXTERNA V2\n")
        archivo.write("=" * 45 + "\n")

        archivo.write(f"Total: {len(registros)}\n")
        archivo.write(f"Aciertos: {aciertos}\n")
        archivo.write(f"Errores: {errores}\n")

        archivo.write(
            f"Exactitud: {exactitud:.4f}\n\n"
        )

        archivo.write(reporte)

    # ======================================================
    # 8. MATRIZ DE CONFUSIÓN
    # ======================================================

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=CLASES_VALIDAS
    )

    visualizacion.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Matriz de confusión - Evaluación externa V2")

    plt.tight_layout()

    plt.savefig(
        RUTA_MATRIZ,
        dpi=300
    )

    plt.show()

    print("\nARCHIVOS GENERADOS")

    print("CSV:", RUTA_CSV)
    print("Métricas:", RUTA_METRICAS)
    print("Matriz:", RUTA_MATRIZ)


# ==========================================================
# 9. EJECUCIÓN
# ==========================================================

if __name__ == "__main__":
    evaluar()
