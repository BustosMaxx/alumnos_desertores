import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos

# configuración del logger
os.makedirs("logs", exist_ok=True)
fecha = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s -%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# Función principal de la aplicación
def main():
    logger.info("Iniciando la aplicación...")

    # Cargar datos
    path = "./data/dataset_TP1_2026.csv"
    data = cargar_datos(path)

    print(data.head())

    # # Preprocesamiento de datos
    # try:
    #     columnas_categoricas = data.select_dtypes(
    #         include=["object", "category"]
    #     ).columns
    #     X = pd.get_dummies(data.drop("target", axis=1), columns=columnas_categoricas, drop_first=True)
    #     y = data["target"]
    #     logger.info("Preprocesamiento de datos completado.")
    # except Exception as e:
    #     logger.error(f"Error en el preprocesamiento de datos: {e}")
    #     return

    logger.info(f">>> Ejecución finalizada. Revisar el archivo {nombre_log} para más detalles.")

if __name__ == "__main__":
    main()