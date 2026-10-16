import logging

from sklearn.metrics import precision_score, recall_score, accuracy_score

logger = logging.getLogger(__name__)

def regresion_logistica(X_train, y_train, X_test, y_test):
    
    """
    Función para entrenar un modelo de regresión logística y evaluar su desempeño.

    Parámetros:
    - X_train: Datos de entrenamiento (características).
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba (características).
    - y_test: Etiquetas de prueba.

    Retorna:
    - model: Modelo de regresión logística entrenado.
    """
    logger.info("Ajustando una regresión logística...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Crear el modelo de regresión logística
    model = LogisticRegression(penalty=None, max_iter=10000)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Hacer predicciones en los datos de prueba
    y_pred = model.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = 2 * (precision * recall) / (precision + recall)

    # Guardar métricas en el logger
    logger.info("Accuracy de la clasificación: {:.2f}%".format(accuracy * 100))
    logger.info("Precision: {:.2f}%".format(precision * 100))
    logger.info("Recall: {:.2f}%".format(recall * 100))
    logger.info("F1-Score: {:.2f}%".format(F1 * 100))
    
    return model

# Support Vector Machine (SVM)
def super_vector_machine(X_train, y_train, X_test, y_test):
    """
    Función para entrenar un modelo de Support Vector Machine (SVM).

    Parámetros:
    - X_train: Datos de entrenamiento (características).
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba (características).
    - y_test: Etiquetas de prueba.

    Retorna:
    - model: Modelo SVM entrenado.
    """
    logger.info("Ajustando un modelo de Support Vector Machine (SVM)...")
    from sklearn.svm import SVC

    # crear el modelo SVM
    model = SVC(kernel='linear', C=1, random_state=42, probability=True)

    # entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # hacer predicciones con los datos de prueba
    y_pred = model.predict(X_test)

    # calcular la precisión de la clasificación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = 2 * (precision * recall) / (precision + recall)

    # Guardar métricas en el logger
    logger.info("Accuracy de la clasificación: {:.2f}%".format(accuracy * 100))
    logger.info("Precision: {:.2f}%".format(precision * 100))
    logger.info("Recall: {:.2f}%".format(recall * 100))
    logger.info("F1-Score: {:.2f}%".format(F1 * 100))

    return model

# Decision Tree
def arbol_decision(X_train, y_train, X_test, y_test):
    """
    Función para entrenar un modelo de Árbol de Decisión.

    Parámetros:
    - X_train: Datos de entrenamiento (características).
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba (características).
    - y_test: Etiquetas de prueba.

    Retorna:
    - model: Modelo de Árbol de Decisión entrenado.
    """
    logger.info("Ajustando un modelo de Árbol de Decisión...")
    from sklearn.tree import DecisionTreeClassifier

    # crear el modelo de Árbol de Decisión
    model = DecisionTreeClassifier(random_state=42)

    # entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # hacer predicciones con los datos de prueba
    y_pred = model.predict(X_test)

    # calcular la precisión de la clasificación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = 2 * (precision * recall) / (precision + recall)

    # Guardar métricas en el logger
    logger.info("Accuracy de la clasificación: {:.2f}%".format(accuracy * 100))
    logger.info("Precision: {:.2f}%".format(precision * 100))
    logger.info("Recall: {:.2f}%".format(recall * 100))
    logger.info("F1-Score: {:.2f}%".format(F1 * 100))

    return model