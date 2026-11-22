
"""
Genera muestras visuales para comparar:
- 12 imágenes originales de Banana.
- 12 imágenes originales de Guava.
- Las 12 imágenes externas clasificadas incorrectamente.

No modifica ni mueve el dataset original.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import random
import csv

# ==========================================================
# 1. RUTAS
# ==========================================================

RAIZ = Path(__file__).resolve().parent.parent

RUTA_TRAIN = RAIZ / "data" / "processed" / "dataset" / "train"
RUTA_ERRORES = RAIZ / "results" / "imagenes_error"

RUTA_SALIDA = RAIZ / "results" / "comparacion_visual"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

CLASES = ["Banana", "Guava"]

EXTENSIONES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CANTIDAD = 12
SEMILLA = 42

# ==========================================================
# 2. OBTENER IMÁGENES
# ==========================================================

def obtener_imagenes(carpeta):

    if not carpeta.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {carpeta}"
        )

    return sorted([
        ruta for ruta in carpeta.rglob("*")
        if ruta.is_file()
        and ruta.suffix.lower() in EXTENSIONES
    ])


def seleccionar_muestra(rutas, cantidad, semilla):

    if len(rutas) < cantidad:
        raise ValueError(
            f"Se necesitan {cantidad} imágenes, "
            f"pero solamente hay {len(rutas)}."
        )

    generador = random.Random(semilla)

    return sorted(
        generador.sample(rutas, cantidad)
    )


# ==========================================================
# 3. CREAR MOSAICO VISUAL
# ==========================================================

def crear_mosaico(rutas, titulo, destino):

    columnas = 4
    filas = 3

    ancho_celda = 300
    alto_celda = 250

    margen = 15
    alto_titulo = 65

    ancho_total = (
        columnas * ancho_celda + (columnas + 1) * margen
    )

    alto_total = (
        alto_titulo
        + filas * alto_celda
        + (filas + 1) * margen
    )

    lienzo = Image.new(
        "RGB",
        (ancho_total, alto_total),
        "#eeeeee"
    )

    dibujo = ImageDraw.Draw(lienzo)

    dibujo.text(
        (margen, 20),
        titulo,
        fill="black"
    )

    for indice, ruta in enumerate(rutas):

        columna = indice % columnas
        fila = indice // columnas

        x = margen + columna * (ancho_celda + margen)
        y = alto_titulo + margen + fila * (
            alto_celda + margen
        )

        try:
            with Image.open(ruta) as original:

                imagen = ImageOps.contain(
                    original.convert("RGB"),
                    (ancho_celda - 20, alto_celda - 55)
                )

            fondo = Image.new(
                "RGB",
                (ancho_celda, alto_celda),
                "white"
            )

            posicion = (
                (ancho_celda - imagen.width) // 2,
                8
            )

            fondo.paste(imagen, posicion)

            lienzo.paste(fondo, (x, y))

            nombre = ruta.name

            dibujo.text(
                (x + 8, y + alto_celda - 30),
                nombre[:38],
                fill="black"
            )

        except Exception as error:
            print(f"Error con {ruta}: {error}")

    lienzo.save(destino)

    print("Mosaico generado:", destino)


# ==========================================================
# 4. PREPARAR MUESTRAS
# ==========================================================

def ejecutar():

    generador = random.Random(SEMILLA)

    muestras = {}

    for clase in CLASES:

        carpeta = RUTA_TRAIN / clase

        imagenes = obtener_imagenes(carpeta)

        muestras[clase] = seleccionar_muestra(
            imagenes,
            CANTIDAD,
            generador.randint(0, 1000000)
        )

        crear_mosaico(
            muestras[clase],
            f"DATASET ORIGINAL - {clase} - 12 IMAGENES",
            RUTA_SALIDA / f"muestra_{clase.lower()}.png"
        )

    # Imágenes externas incorrectas
    errores = obtener_imagenes(RUTA_ERRORES)

    if len(errores) != 12:
        raise ValueError(
            f"Se esperaban 12 imágenes de error, "
            f"pero se encontraron {len(errores)}."
        )

    crear_mosaico(
        errores,
        "IMAGENES EXTERNAS CLASIFICADAS INCORRECTAMENTE",
        RUTA_SALIDA / "muestra_errores_externos.png"
    )

    # ======================================================
    # 5. CREAR MANIFIESTO DE LA MUESTRA
    # ======================================================

    ruta_csv = RUTA_SALIDA / "manifiesto_muestra.csv"

    with open(
        ruta_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        campos = ["grupo", "clase", "imagen", "ruta"]

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()

        for clase, rutas in muestras.items():

            for ruta in rutas:

                escritor.writerow({
                    "grupo": "dataset_original",
                    "clase": clase,
                    "imagen": ruta.name,
                    "ruta": str(ruta.relative_to(RAIZ))
                })

        for ruta in errores:

            escritor.writerow({
                "grupo": "error_externo",
                "clase": ruta.stem.split("_")[0],
                "imagen": ruta.name,
                "ruta": str(ruta.relative_to(RAIZ))
            })

    print("\nCOMPARACIÓN VISUAL COMPLETADA")
    print("=" * 50)
    print("Banana originales: 12")
    print("Guava originales: 12")
    print("Errores externos: 12")
    print("Total: 36 imágenes")
    print("Resultados:", RUTA_SALIDA)


if __name__ == "__main__":
    ejecutar()
