import pandas as pd
import os
import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.loader import cargar_datos
from src.preprocesamiento import procesar
from src.machine_learning import regresion_logistica, super_vector_machine, arbol_decision
from src.optimizacion_hp import optimizar_hiperparametros_GS, optimizar_hiperparametros_RS

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

    # Preprocesamiento de datos
    X = procesar(data)
    y = data["target"]
    # dividir los datos en conjunto de entrenamiento y prueba 80/20 estratificado
    logger.info("Dividiendo los datos en conjuntos de entrenamiento y prueba (80/20 estratificado).")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)

    # Entrenar y evaluar el modelo
    reg_log = regresion_logistica(X_train, y_train, X_test, y_test)
    svm = super_vector_machine(X_train, y_train, X_test, y_test)
    desicion_tree = arbol_decision(X_train, y_train, X_test, y_test)

    # Guardar los modelos entrenados
    logger.info("Guardando los modelos entrenados...")
    os.makedirs("modelos", exist_ok=True)
    joblib.dump(reg_log, "modelos/modelo_regresion_logistica.pkl")
    joblib.dump(svm, "modelos/modelo_svm.pkl")
    joblib.dump(desicion_tree, "modelos/modelo_arbol_decision.pkl")

    # Optimización de hiperparámetros (opcional)
    logger.info("Iniciando la optimización de hiperparámetros para la regresión logística...")
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
    reg_log_optimized = optimizar_hiperparametros_GS(reg_log, param_grid, X_train, y_train)
    joblib.dump(reg_log_optimized, "modelos/modelo_regresion_logistica_optimizado.pkl")
    logger.info("Optimización de hiperparámetros para la regresión logística finalizada.")
    
    #Predecir sobre el conjunto de test
    y_pred = reg_log_optimized.predict(X_test)

    # Calcular métricas
    logger.info("Calculando métricas...")
    logger.info(classification_report(y_test, y_pred))



    logger.info(f">>> Ejecución finalizada. Revisar el archivo {nombre_log} para más detalles.")

if __name__ == "__main__":
    main()