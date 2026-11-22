
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import math

RAIZ = Path(__file__).resolve().parent.parent

RUTA_CSV = RAIZ / "results" / "evaluacion_externa_v2.csv"
RUTA_SALIDA = RAIZ / "results" / "comparacion_visual" / "errores_v2.png"

ANCHO = 360
ALTO = 300
COLUMNAS = 3

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

errores = []

with open(RUTA_CSV, encoding="utf-8-sig") as archivo:
    lector = csv.DictReader(archivo)

    for registro in lector:
        if registro["correcta"].lower() == "false":
            errores.append(registro)

filas = math.ceil(len(errores) / COLUMNAS)

mosaico = Image.new(
    "RGB",
    (COLUMNAS * ANCHO, filas * ALTO),
    "white"
)

dibujo = ImageDraw.Draw(mosaico)

for i, registro in enumerate(errores):

    ruta = RAIZ / registro["imagen"]

    imagen = Image.open(ruta).convert("RGB")
    imagen.thumbnail((ANCHO - 30, 220))

    columna = i % COLUMNAS
    fila = i // COLUMNAS

    x = columna * ANCHO
    y = fila * ALTO

    posicion_x = x + (ANCHO - imagen.width) // 2

    mosaico.paste(imagen, (posicion_x, y + 10))

    dibujo.text(
        (x + 10, y + 238),
        f"Real: {registro['etiqueta_real']}",
        fill="black"
    )

    dibujo.text(
        (x + 10, y + 255),
        f"Predicción: {registro['prediccion']}",
        fill="red"
    )

    dibujo.text(
        (x + 10, y + 272),
        f"Confianza: {float(registro['confianza']):.2%}",
        fill="black"
    )

mosaico.save(RUTA_SALIDA)

print("Mosaico generado:", RUTA_SALIDA)
