
from pathlib import Path
from collections import defaultdict
import hashlib

# ==================================================
# 1. CONFIGURACIÓN
# ==================================================

RAIZ = Path(__file__).resolve().parent.parent

DATASET = RAIZ / "data" / "processed" / "dataset"

PARTICIONES = ["train", "val", "test"]
CLASES = ["Banana", "Guava"]

EXTENSIONES = {".jpg", ".jpeg", ".png"}

# Diccionario: hash -> información de las imágenes
imagenes_por_hash = defaultdict(list)

# ==================================================
# 2. FUNCIÓN SHA-256
# ==================================================

def calcular_sha256(ruta):
    sha256 = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


# ==================================================
# 3. RECORRER LAS PARTICIONES
# ==================================================

print("=" * 60)
print("COMPROBACIÓN DE DUPLICADOS ENTRE PARTICIONES")
print("=" * 60)

total_imagenes = 0

for particion in PARTICIONES:

    print(f"\nRevisando: {particion.upper()}")

    for clase in CLASES:

        carpeta = DATASET / particion / clase

        if not carpeta.exists():
            print(f"ERROR: No existe {carpeta}")
            continue

        imagenes = [
            p for p in carpeta.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        ]

        print(f"  {clase}: {len(imagenes)} imágenes")

        for ruta in imagenes:

            hash_imagen = calcular_sha256(ruta)

            imagenes_por_hash[hash_imagen].append({
                "ruta": ruta,
                "particion": particion,
                "clase": clase
            })

            total_imagenes += 1


# ==================================================
# 4. IDENTIFICAR HASHES REPETIDOS
# ==================================================

duplicados_mismo_conjunto = []
duplicados_entre_conjuntos = []
duplicados_entre_clases = []

for hash_imagen, registros in imagenes_por_hash.items():

    if len(registros) > 1:

        particiones = {
            registro["particion"]
            for registro in registros
        }

        clases = {
            registro["clase"]
            for registro in registros
        }

        if len(particiones) > 1:
            duplicados_entre_conjuntos.append(
                (hash_imagen, registros)
            )
        else:
            duplicados_mismo_conjunto.append(
                (hash_imagen, registros)
            )

        if len(clases) > 1:
            duplicados_entre_clases.append(
                (hash_imagen, registros)
            )


# ==================================================
# 5. INFORME FINAL
# ==================================================

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Total de imágenes examinadas: {total_imagenes}")
print(f"Hashes únicos: {len(imagenes_por_hash)}")

print(
    "Duplicados entre train, val y test:",
    len(duplicados_entre_conjuntos)
)

print(
    "Duplicados dentro de una misma partición:",
    len(duplicados_mismo_conjunto)
)

print(
    "Duplicados entre clases:",
    len(duplicados_entre_clases)
)

# Mostrar detalles si existen duplicados entre particiones
if duplicados_entre_conjuntos:

    print("\nADVERTENCIA: Se encontraron duplicados entre particiones.")

    for hash_imagen, registros in duplicados_entre_conjuntos:

        print(f"\nSHA-256: {hash_imagen}")

        for registro in registros:
            print(
                registro["particion"],
                registro["clase"],
                registro["ruta"]
            )

else:

    print("\nRESULTADO: No existen imágenes idénticas")
    print("compartidas entre entrenamiento, validación y prueba.")

# ==================================================
# 6. DICTAMEN
# ==================================================

print("\n" + "=" * 60)

if total_imagenes != 1272:
    print("DICTAMEN: Revisar el número total de imágenes.")

elif duplicados_entre_conjuntos:
    print("DICTAMEN: NO APTO. Existe fuga de datos.")

elif duplicados_mismo_conjunto:
    print("DICTAMEN: REVISAR. Hay duplicados dentro de particiones.")

elif duplicados_entre_clases:
    print("DICTAMEN: REVISAR. Hay imágenes repetidas entre clases.")

else:
    print("DICTAMEN: APTO PARA CONTINUAR CON EL ENTRENAMIENTO.")

print("=" * 60)
