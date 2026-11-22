
"""
Evaluación interna del modelo Banana - Guava.
Utiliza exclusivamente el conjunto test.
"""

from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader
from torchvision import datasets

from src.predict import cargar_modelo
from src.preprocessing import obtener_transformaciones


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

RAIZ = Path(__file__).resolve().parent.parent

RUTA_TEST = RAIZ / "data" / "processed" / "dataset" / "test"
RUTA_RESULTADOS = RAIZ / "results"

RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

CLASES_VALIDAS = ["Banana", "Guava"]

TAMANO_LOTE = 16

DISPOSITIVO = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# 2. CARGAR DATASET
# ==========================================================

def cargar_dataset():

    if not RUTA_TEST.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {RUTA_TEST}"
        )

    transformaciones = obtener_transformaciones()

    dataset = datasets.ImageFolder(
        root=str(RUTA_TEST),
        transform=transformaciones["test"]
    )

    if dataset.classes != CLASES_VALIDAS:
        raise ValueError(
            f"Orden de clases inesperado: {dataset.classes}"
        )

    cargador = DataLoader(
        dataset,
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0
    )

    return dataset, cargador


# ==========================================================
# 3. EVALUACIÓN DEL MODELO
# ==========================================================

def evaluar():

    modelo, clases = cargar_modelo()

    modelo = modelo.to(DISPOSITIVO)
    modelo.eval()

    dataset, cargador = cargar_dataset()

    etiquetas_reales = []
    predicciones = []
    registros = []

    print("\nEVALUACIÓN INTERNA DEL MODELO")
    print("=" * 55)
    print("Dataset:", RUTA_TEST)
    print("Imágenes:", len(dataset))
    print("Clases:", clases)
    print("Dispositivo:", DISPOSITIVO)

    with torch.inference_mode():

        for imagenes, etiquetas in cargador:

            imagenes = imagenes.to(DISPOSITIVO)

            salidas = modelo(imagenes)

            probabilidades = torch.softmax(
                salidas,
                dim=1
            )

            indices = probabilidades.argmax(dim=1)

            for i in range(len(etiquetas)):

                etiqueta_real = clases[etiquetas[i].item()]
                prediccion = clases[indices[i].item()]

                confianza = probabilidades[
                    i, indices[i]
                ].item()

                etiquetas_reales.append(etiqueta_real)
                predicciones.append(prediccion)

                ruta = dataset.samples[
                    len(registros)
                ][0]

                registros.append({
                    "imagen": str(ruta),
                    "etiqueta_real": etiqueta_real,
                    "prediccion": prediccion,
                    "confianza": round(confianza, 6),
                    "correcta": etiqueta_real == prediccion
                })

    # ======================================================
    # 4. MÉTRICAS
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
        registro["correcta"] for registro in registros
    )

    errores = len(registros) - aciertos

    print("\nRESUMEN GENERAL")
    print("=" * 55)
    print("Total:", len(registros))
    print("Aciertos:", aciertos)
    print("Errores:", errores)
    print(f"Exactitud: {exactitud:.2%}")

    print("\nREPORTE DE CLASIFICACIÓN")
    print(reporte)

    # ======================================================
    # 5. EXPORTAR PREDICCIONES
    # ======================================================

    ruta_csv = RUTA_RESULTADOS / "evaluacion_interna.csv"

    with open(
        ruta_csv,
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
    # 6. EXPORTAR MÉTRICAS
    # ======================================================

    ruta_metricas = (
        RUTA_RESULTADOS / "metricas_internas.txt"
    )

    with open(
        ruta_metricas,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write("EVALUACIÓN INTERNA\n")
        archivo.write("=" * 40 + "\n")
        archivo.write(f"Total: {len(registros)}\n")
        archivo.write(f"Aciertos: {aciertos}\n")
        archivo.write(f"Errores: {errores}\n")
        archivo.write(f"Exactitud: {exactitud:.4f}\n\n")
        archivo.write(reporte)

    # ======================================================
    # 7. MATRIZ DE CONFUSIÓN
    # ======================================================

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=CLASES_VALIDAS
    )

    visualizacion.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Matriz de confusión - Evaluación interna")
    plt.tight_layout()

    plt.savefig(
        RUTA_RESULTADOS / "matriz_confusion_interna.png",
        dpi=300
    )

    plt.show()

    print("\nARCHIVOS GENERADOS")
    print("CSV:", ruta_csv)
    print("Métricas:", ruta_metricas)
    print(
        "Matriz:",
        RUTA_RESULTADOS / "matriz_confusion_interna.png"
    )


if __name__ == "__main__":
    evaluar()
