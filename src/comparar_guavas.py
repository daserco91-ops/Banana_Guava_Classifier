
from pathlib import Path
from PIL import Image, ImageOps

RUTA_1 = Path("data/external_test/Guava/Guava_007.png")
RUTA_2 = Path("data/external_test/Guava/Guava_010.png")


def calcular_hash_diferencial(ruta):
    imagen = Image.open(ruta).convert("L")
    imagen = ImageOps.fit(imagen, (9, 8))
    pixeles = list(imagen.get_flattened_data())

    bits = []

    for fila in range(8):
        for columna in range(8):
            izquierda = pixeles[fila * 9 + columna]
            derecha = pixeles[fila * 9 + columna + 1]
            bits.append(izquierda > derecha)

    return bits


def comparar_hashes(hash1, hash2):
    distancia = sum(a != b for a, b in zip(hash1, hash2))
    similitud = (1 - distancia / 64) * 100

    return distancia, similitud


if __name__ == "__main__":
    hash1 = calcular_hash_diferencial(RUTA_1)
    hash2 = calcular_hash_diferencial(RUTA_2)

    distancia, similitud = comparar_hashes(hash1, hash2)

    print("=" * 45)
    print("COMPARACIÓN DE GUAYABAS")
    print("=" * 45)
    print("Imagen 1:", RUTA_1.name)
    print("Imagen 2:", RUTA_2.name)
    print("Distancia Hamming:", distancia, "/ 64")
    print(f"Similitud dHash: {similitud:.2f}%")
