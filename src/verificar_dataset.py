
from pathlib import Path

# Ruta principal del proyecto
RAIZ = Path(__file__).resolve().parent.parent

# Ruta del dataset procesado
DATASET = RAIZ / "data" / "processed" / "dataset"

particiones = ["train", "val", "test"]
clases = ["Banana", "Guava"]

total = 0

print("VERIFICACIÓN DEL DATASET")
print("=" * 45)

for particion in particiones:
    print(f"\n{particion.upper()}")

    for clase in clases:
        carpeta = DATASET / particion / clase

        if not carpeta.exists():
            print(f"ERROR: No existe {carpeta}")
            continue

        imagenes = [
            p for p in carpeta.iterdir()
            if p.is_file() and p.suffix.lower() in
            [".jpg", ".jpeg", ".png"]
        ]

        cantidad = len(imagenes)
        total += cantidad

        print(f"{clase}: {cantidad} imágenes")

print("\n" + "=" * 45)

print(f"TOTAL DE IMÁGENES: {total}")

if total == 1272:
    print("RESULTADO: Dataset completo y localizado.")
else:
    print("RESULTADO: Revisar archivos y ubicaciones.")
