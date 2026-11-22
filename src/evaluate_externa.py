
"""Evaluación externa del clasificador Banana_Guava_Classifier."""

from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.predict import cargar_modelo
from src.preprocessing import preparar_imagen


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

RUTA_DATASET = Path("data/external_test")
RUTA_RESULTADOS = Path("results")
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASES_VALIDAS = ["Banana", "Guava"]

DISPOSITIVO = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# 2. OBTENER IMÁGENES Y ETIQUETAS
# ==========================================================

def obtener_imagenes():
    imagenes = []

    for carpeta in RUTA_DATASET.iterdir():
        if not carpeta.is_dir():
            continue

        nombre = carpeta.name.lower()

        if nombre == "banana":
            etiqueta = "Banana"
        elif nombre == "guava":
            etiqueta = "Guava"
        else:
            print(f"Carpeta ignorada: {carpeta.name}")
            continue

        for ruta in carpeta.rglob("*"):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES:
                imagenes.append((ruta, etiqueta))

    return sorted(imagenes, key=lambda elemento: str(elemento[0]))


# ==========================================================
# 3. EVALUACIÓN
# ==========================================================

def evaluar():
    modelo, clases = cargar_modelo()

    if clases != CLASES_VALIDAS:
        raise ValueError(
            f"Orden de clases inesperado: {clases}"
        )

    imagenes = obtener_imagenes()

    if not imagenes:
        raise RuntimeError("No se encontraron imágenes externas.")

    etiquetas_reales = []
    predicciones = []
    registros = []

    print("\nEVALUACIÓN DEL MODELO")
    print("=" * 55)
    print("Imágenes encontradas:", len(imagenes))
    print("Dispositivo:", DISPOSITIVO)

    for numero, (ruta, etiqueta_real) in enumerate(imagenes, start=1):

        tensor = preparar_imagen(ruta).to(DISPOSITIVO)

        with torch.inference_mode():
            salidas = modelo(tensor)
            probabilidades = torch.softmax(salidas, dim=1)[0]

        indice = probabilidades.argmax().item()
        prediccion = clases[indice]
        confianza = probabilidades[indice].item()

        etiquetas_reales.append(etiqueta_real)
        predicciones.append(prediccion)

        registros.append({
            "imagen": str(ruta),
            "etiqueta_real": etiqueta_real,
            "prediccion": prediccion,
            "confianza": round(confianza, 6),
            "correcta": etiqueta_real == prediccion
        })

        print(
            f"[{numero:02d}/{len(imagenes)}] "
            f"Real: {etiqueta_real:7} | "
            f"Predicción: {prediccion:7} | "
            f"Confianza: {confianza:.2%}"
        )

    # ------------------------------------------------------
    # 4. MÉTRICAS
    # ------------------------------------------------------

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
    print("Total de imágenes:", len(registros))
    print("Aciertos:", aciertos)
    print("Errores:", errores)
    print(f"Exactitud: {exactitud:.2%}")

    print("\nREPORTE DE CLASIFICACIÓN")
    print(reporte)

    # ------------------------------------------------------
    # 5. EXPORTAR PREDICCIONES
    # ------------------------------------------------------

    ruta_csv = RUTA_RESULTADOS / "evaluacion_externa.csv"

    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as archivo:
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

    # ------------------------------------------------------
    # 6. EXPORTAR MÉTRICAS
    # ------------------------------------------------------

    ruta_metricas = RUTA_RESULTADOS / "metricas_evaluacion.txt"

    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        archivo.write("EVALUACIÓN EXTERNA\n")
        archivo.write("=" * 40 + "\n")
        archivo.write(f"Total: {len(registros)}\n")
        archivo.write(f"Aciertos: {aciertos}\n")
        archivo.write(f"Errores: {errores}\n")
        archivo.write(f"Exactitud: {exactitud:.4f}\n\n")
        archivo.write(reporte)

    # ------------------------------------------------------
    # 7. MATRIZ DE CONFUSIÓN
    # ------------------------------------------------------

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=CLASES_VALIDAS
    )

    visualizacion.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Matriz de confusión - Evaluación externa")
    plt.tight_layout()

    plt.savefig(
        RUTA_RESULTADOS / "matriz_confusion.png",
        dpi=300
    )

    plt.show()

    print("\nARCHIVOS GENERADOS")
    print("CSV:", ruta_csv)
    print("Métricas:", ruta_metricas)
    print("Matriz:", RUTA_RESULTADOS / "matriz_confusion.png")


# ==========================================================
# 8. EJECUCIÓN
# ==========================================================

if __name__ == "__main__":
    evaluar()
