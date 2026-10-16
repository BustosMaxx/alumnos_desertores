import logging

logger = logging.getLogger(__name__)

def optimizar_hiperparametros_GS(model, param_grid, X_train, y_train):
    """
    Función para optimizar los hiperparámetros de un modelo utilizando GridSearchCV.

    Parámetros:
    - model: Modelo de aprendizaje automático a optimizar.
    - param_grid: Diccionario con los hiperparámetros a probar.
    - X_train: Datos de entrenamiento (características).
    - y_train: Etiquetas de entrenamiento.

    Retorna:
    - best_model: Modelo con los mejores hiperparámetros encontrados.
    """
    from sklearn.model_selection import GridSearchCV

    logger.info("Iniciando la optimización de hiperparámetros...")
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='f1', cv=5, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo encontrado
    best_model = grid_search.best_estimator_
    # Guardar los mejores hiperparámetros encontrados en el logger
    logger.info(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    
    return best_model

def optimizar_hiperparametros_RS(model, param_distributions, X_train, y_train, n_iter=10):
    """
    Función para optimizar los hiperparámetros de un modelo utilizando RandomizedSearchCV.

    Parámetros:
    - model: Modelo de aprendizaje automático a optimizar.
    - param_distributions: Diccionario con las distribuciones de los hiperparámetros a probar.
    - X_train: Datos de entrenamiento (características).
    - y_train: Etiquetas de entrenamiento.
    - n_iter: Número de iteraciones aleatorias a realizar.

    Retorna:
    - best_model: Modelo con los mejores hiperparámetros encontrados.
    """
    from sklearn.model_selection import RandomizedSearchCV

    logger.info("Iniciando la optimización de hiperparámetros con búsqueda aleatoria...")
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                        n_iter=n_iter, scoring='f1', cv=5, n_jobs=-1, random_state=42)
    
    random_search.fit(X_train, y_train)

    # Obtener el mejor modelo encontrado
    best_model = random_search.best_estimator_
    # Guardar los mejores hiperparámetros encontrados en el logger
    logger.info(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")
    
    return best_model