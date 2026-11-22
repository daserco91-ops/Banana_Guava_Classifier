
from pathlib import Path
import shutil

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset"

particiones = ["train", "val", "test"]
clases = ["Banana", "Guava"]

extensiones = {".jpg", ".jpeg", ".png"}

print("ORGANIZANDO DATASET")
print("=" * 45)

for particion in particiones:

    for clase in clases:

        nombre_origen = (
            f"{particion}_{clase.lower()}_banana_guava"
        )

        origen = DATASET / particion / nombre_origen
        destino = DATASET / particion / clase

        destino.mkdir(parents=True, exist_ok=True)

        if not origen.exists():
            print(f"No encontrada: {origen}")
            continue

        imagenes = [
            p for p in origen.rglob("*")
            if p.is_file() and p.suffix.lower() in extensiones
        ]

        copiadas = 0

        for imagen in imagenes:

            archivo_destino = destino / imagen.name

            if archivo_destino.exists():
                print(f"Archivo ya existente: {imagen.name}")
                continue

            shutil.copy2(imagen, archivo_destino)
            copiadas += 1

        print(
            f"{particion} - {clase}: "
            f"{copiadas} imágenes copiadas"
        )

print("\nOrganización terminada.")
