import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def procesar(data):
    """ Limpiar y procesar"""
    logger.info("Iniciando el procesamiento de los datos.")

    # Seleccionar filas donde 'target' es 1
    data["target"] = np.where(data['target'] == "Desertor", 1, 0)

    # OneHotEnconding
    columnas_categoricas = data.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(data.drop("target", axis=1), columns=columnas_categoricas, drop_first=True)
    logger.info(f"Dimensiones del dataset original: {data.shape}")
    logger.info(f"Shape of the transformed feature matrix X: {X.shape}")

    return X
    



    

    