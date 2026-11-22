
"""
Simulación de inferencia en tiempo real.
Banana_Guava_Classifier
"""

from pathlib import Path
import time
import csv

from PIL import Image
import torch

from src.predict import cargar_modelo
from src.preprocessing import preparar_imagen


# ==================================================
# CONFIGURACIÓN
# ==================================================

RAIZ = Path(__file__).resolve().parent.parent

RUTA_IMAGENES = RAIZ / "data" / "external_test"
RUTA_RESULTADOS = RAIZ / "results"

RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device("cpu")
PAUSA_ENTRE_IMAGENES = 0.5


# ==================================================
# CARGA DEL MODELO
# ==================================================

def iniciar_modelo():

    modelo, clases = cargar_modelo()

    modelo.to(DISPOSITIVO)
    modelo.eval()

    print("Modelo cargado correctamente.")
    print("Clases:", clases)

    return modelo, clases


# ==================================================
# PROCESAMIENTO DE UNA IMAGEN
# ==================================================

def procesar_imagen(modelo, clases, ruta):

    inicio = time.perf_counter()

    tensor = preparar_imagen(ruta)
    tensor = tensor.to(DISPOSITIVO)

    with torch.inference_mode():
        salida = modelo(tensor)
        probabilidades = torch.softmax(salida, dim=1)[0]

        indice = probabilidades.argmax().item()
        clase = clases[indice]
        confianza = probabilidades[indice].item()

    tiempo = time.perf_counter() - inicio

    return {
        "imagen": ruta.name,
        "clase_predicha": clase,
        "confianza": confianza,
        "tiempo_segundos": tiempo
    }


# ==================================================
# SIMULACIÓN DE FLUJO CONTINUO
# ==================================================

def obtener_imagenes():

    extensiones = {".jpg", ".jpeg", ".png", ".webp"}

    imagenes = sorted(
        ruta
        for ruta in RUTA_IMAGENES.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in extensiones
    )

    return imagenes


def ejecutar_simulacion():

    modelo, clases = iniciar_modelo()
    imagenes = obtener_imagenes()

    if not imagenes:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {RUTA_IMAGENES}"
        )

    resultados = []

    print("\nINICIO DE SIMULACIÓN")
    print("Imágenes:", len(imagenes))
    print("-" * 55)

    inicio_total = time.perf_counter()

    for numero, ruta in enumerate(imagenes, start=1):

        resultado = procesar_imagen(modelo, clases, ruta)
        resultados.append(resultado)

        print(
            f"{numero}/{len(imagenes)} | "
            f"{resultado['imagen']} | "
            f"{resultado['clase_predicha']} | "
            f"{resultado['confianza']:.2%} | "
            f"{resultado['tiempo_segundos']:.4f} s"
        )

        time.sleep(PAUSA_ENTRE_IMAGENES)

    tiempo_total = time.perf_counter() - inicio_total

    fps = len(resultados) / tiempo_total

    # ==============================================
    # GUARDAR RESULTADOS
    # ==============================================

    archivo = RUTA_RESULTADOS / "inferencia_tiempo_real.csv"

    with open(
        archivo,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        campos = [
            "imagen",
            "clase_predicha",
            "confianza",
            "tiempo_segundos"
        ]

        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)

    # ==============================================
    # RESUMEN
    # ==============================================

    promedio = sum(
        r["tiempo_segundos"] for r in resultados
    ) / len(resultados)

    print("\nRESUMEN DE SIMULACIÓN")
    print("-" * 55)
    print("Imágenes procesadas:", len(resultados))
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    print(f"Tiempo promedio por imagen: {promedio:.4f} segundos")
    print(f"Velocidad aproximada: {fps:.2f} imágenes/segundo")
    print("Resultados guardados en:", archivo)


if __name__ == "__main__":
    ejecutar_simulacion()
