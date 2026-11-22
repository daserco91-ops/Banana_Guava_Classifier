
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.predict import cargar_modelo
from src.preprocessing import preparar_imagen

st.set_page_config(
    page_title="Clasificador Banana y Guava",
    page_icon="🍌",
    layout="centered"
)

st.title("🍌 Clasificador de frutas")
st.write("Carga una imagen para identificar si contiene Banana o Guava.")

@st.cache_resource
def cargar_modelo_app():
    return cargar_modelo()

modelo, clases = cargar_modelo_app()

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"]
)

if archivo is not None:

    imagen = Image.open(archivo).convert("RGB")

    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )

    if st.button("Analizar imagen", type="primary"):

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temporal:

            imagen.save(temporal.name)
            ruta_temporal = Path(temporal.name)

        try:
            tensor = preparar_imagen(ruta_temporal)

            import torch

            with torch.inference_mode():
                salida = modelo(tensor)
                probabilidades = torch.softmax(
                    salida, dim=1
                )[0]

            indice = probabilidades.argmax().item()
            clase = clases[indice]
            confianza = probabilidades[indice].item() * 100

            st.success(f"Clasificación: {clase}")

            st.metric(
                "Confianza del modelo",
                f"{confianza:.2f}%"
            )

            st.subheader("Probabilidades por clase")

            for i, nombre in enumerate(clases):
                porcentaje = probabilidades[i].item() * 100

                st.write(f"{nombre}: {porcentaje:.2f}%")
                st.progress(porcentaje / 100)

        finally:
            ruta_temporal.unlink(missing_ok=True)
