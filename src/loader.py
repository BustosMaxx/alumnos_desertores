import pandas as pd
import logging

logger = logging.getLogger(__name__)

## función cargar datos
def cargar_datos(ruta_archivo):
    # Cargar datos
    logger.info(f"Cargando datos desde path: {ruta_archivo} ...")
    try:
        data = pd.read_csv(ruta_archivo)
        logger.info(f"Datos cargados correctamente. Dimensiones del dataset: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        raise