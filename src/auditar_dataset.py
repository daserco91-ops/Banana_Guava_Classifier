
from pathlib import Path
from PIL import Image
import csv

# ==========================================
# CONFIGURACIÓN
# ==========================================

RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "raw"
RUTA_RESULTADOS = Path(__file__).resolve().parent.parent / "results"

CLASES = ["Banana", "Guava"]
FORMATOS_VALIDOS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==========================================
# AUDITORÍA DEL DATASET
# ==========================================

def auditar_dataset():

    RUTA_RESULTADOS.mkdir(exist_ok=True)

    total_imagenes = 0
    imagenes_danadas = []
    registros = []

    print("\nAUDITORÍA DEL DATASET")
    print("=" * 45)

    for clase in CLASES:

        carpeta = RUTA_DATASET / clase

        if not carpeta.exists():
            print(f"ERROR: No existe la carpeta {clase}")
            continue

        archivos = [
            archivo for archivo in carpeta.rglob("*")
            if archivo.is_file()
            and archivo.suffix.lower() in FORMATOS_VALIDOS
        ]

        validas = 0

        for archivo in archivos:
            try:
                with Image.open(archivo) as imagen:
                    imagen.verify()

                with Image.open(archivo) as imagen:
                    ancho, alto = imagen.size
                    formato = imagen.format

                registros.append({
                    "archivo": archivo.name,
                    "clase": clase,
                    "formato": formato,
                    "ancho": ancho,
                    "alto": alto,
                    "estado": "Valida"
                })

                validas += 1

            except Exception as error:
                imagenes_danadas.append(str(archivo))

                registros.append({
                    "archivo": archivo.name,
                    "clase": clase,
                    "formato": "",
                    "ancho": "",
                    "alto": "",
                    "estado": f"Dañada: {error}"
                })

        total_imagenes += len(archivos)

        print(f"{clase}:")
        print(f"  Encontradas: {len(archivos)}")
        print(f"  Válidas: {validas}")
        print(f"  Dañadas: {len(archivos) - validas}")

    # ======================================
    # GUARDAR INFORME CSV
    # ======================================

    ruta_csv = RUTA_RESULTADOS / "auditoria_dataset.csv"

    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as archivo:
        campos = [
            "archivo", "clase", "formato",
            "ancho", "alto", "estado"
        ]

        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)

    print("\nRESUMEN GENERAL")
    print("=" * 45)
    print(f"Total de imágenes: {total_imagenes}")
    print(f"Imágenes dañadas: {len(imagenes_danadas)}")
    print(f"Informe generado: {ruta_csv}")

if __name__ == "__main__":
    auditar_dataset()

