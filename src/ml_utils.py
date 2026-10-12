"""Reusable helpers for classical ML workflows (scikit-learn).

Train/evaluate wrappers (RandomForest, KNN, LinearSVC, DecisionTree, AdaBoost,
GradientBoosting, MLP), grid-search variants, plotting helpers and PCA/K-Means utilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def plot_correlation_matrix(data):
    """
    Plot a correlation matrix for all features in the dataset.

    Parameters:
    data (DataFrame): The input DataFrame containing all features.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar=True)
    plt.title("Correlation Matrix")
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance of a trained model.

    Parameters:
    model: The trained model with feature_importances_ attribute (e.g., RandomForest).
    feature_names (list): List of feature names used in the model.

    Returns:
    None
    """
    # Get feature importances and sort them
    importance_values = model.feature_importances_
    sorted_indices = np.argsort(importance_values)[::-1]
    sorted_importances = importance_values[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance Ranking")
    plt.gca().invert_yaxis()
    plt.show()


def plot_feature_importance_by_correlation(data, target_column):
    """
    Calculate the correlation between each feature and the target variable and plot the feature importances.
    
    Parameters:
    - data (DataFrame): The input DataFrame including both features and the target column.
    - target_column (str): The name of the target column.
    
    Returns:
    - None: Displays a bar plot of feature importances based on correlation.
    """
    # Calculating the correlation between each feature and the target
    correlations = data.corr()[target_column].drop(target_column)  # Drop the target from correlation

    # Create a DataFrame to hold the feature importances
    feature_importances = pd.DataFrame({'Feature': correlations.index, 'Importance': correlations.values})
    
    # Sort features by importance (absolute correlation value)
    feature_importances['Importance'] = feature_importances['Importance'].abs()  # Absolute value of correlation
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importances Based on Correlation with Target')
    plt.xlabel('Importance (Correlation with Target)')
    plt.ylabel('Feature')
    plt.show()


def standardize_features(data, features):
    """
    Estandariza características específicas del dataset.
    
    Parameters:
    - data: DataFrame con los datos.
    - features: Lista de características a estandarizar.
    
    Returns:
    - DataFrame con las características seleccionadas estandarizadas.
    """
    # Copia del dataset para no modificar el original
    data_standardized = data.copy()
    
    # Crear el escalador y aplicar a las características seleccionadas
    scaler = StandardScaler()
    data_standardized[features] = scaler.fit_transform(data[features])
    
    return data_standardized


def train_random_forest_with_split(data, target_column, test_size=0.2, random_state=42):
    """
    Train a RandomForestClassifier with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.

    Returns:
    model: Trained RandomForestClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separar las características y el objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Entrenar el modelo
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_random_forest(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo RandomForestClassifier con el conjunto de datos de entrenamiento y evalúa el modelo
    con el conjunto de datos de prueba.

    Parameters:
    - data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    - Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    - target_column (str): Nombre de la columna objetivo. Default es 'survived'.
    - random_state (int): Semilla para la reproducibilidad.

    Returns:
    - model: Modelo RandomForestClassifier entrenado.
    - accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo del conjunto de datos de entrenamiento
    X_train = data.drop(target_column, axis=1)  # Eliminar la columna objetivo
    y_train = data[target_column]  # Columna objetivo

    # Definir el modelo de Random Forest con hiperparámetros especificados
    rf_model = RandomForestClassifier(
        n_estimators=200,       # Número de árboles en el bosque
        max_depth=20,           # Profundidad máxima del árbol
        min_samples_split=5,    # Mínimo de muestras requeridas para dividir un nodo interno
        min_samples_leaf=2,     # Mínimo de muestras requeridas para estar en un nodo hoja
        bootstrap=True,         # Si se utilizan muestras bootstrap al construir los árboles
        random_state=random_state
    )

    # Entrenar el modelo
    rf_model.fit(X_train, y_train)

    # Separar las características y el objetivo del conjunto de prueba
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')  # Eliminar la columna objetivo
    y_test = Tdata[target_column]  # Columna objetivo

    # Hacer predicciones sobre el conjunto de prueba
    y_pred = rf_model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return rf_model, accuracy


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on a given model and dataset.
    
    Parameters:
    model (estimator): The machine learning model to evaluate.
    X (DataFrame): Feature set.
    y (Series): Target variable.
    cv (int): Number of cross-validation folds.
    scoring (str): Scoring metric for cross-validation.
    
    Returns:
    cv_scores (array): Array of cross-validation scores.
    mean_score (float): Average accuracy across all folds.
    """
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Display results
    print("Cross-validation scores for each fold:", cv_scores)
    print(f'Average {scoring} after cross-validation: {cv_scores.mean():.2f}')
    
    return cv_scores, cv_scores.mean()


def train_knn_with_split(data, target_column, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Train a KNeighborsClassifier with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    n_neighbors (int): Number of neighbors for KNN.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.

    Returns:
    model: Trained KNeighborsClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_knn(data, Tdata, target_column, n_neighbors=5):
    """
    Train a KNeighborsClassifier model with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - n_neighbors (int): Number of neighbors for KNN.

    Returns:
    - model: Trained KNeighborsClassifier model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model
    knn_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = knn_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return knn_model, accuracy


def train_linear_svc_with_split(data, target_column, test_size=0.2, random_state=42, max_iter=1000):
    """
    Train a LinearSVC model with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.
    max_iter (int): Maximum number of iterations for the model.

    Returns:
    model: Trained LinearSVC model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the LinearSVC model
    model = LinearSVC(random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_linear_svc(data, Tdata, target_column, random_state=42, max_iter=1000):
    """
    Train a LinearSVC model with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - random_state (int): Seed for reproducibility.
    - max_iter (int): Maximum number of iterations for the model.

    Returns:
    - model: Trained LinearSVC model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the LinearSVC model
    svc_model = LinearSVC(random_state=random_state, max_iter=max_iter)

    # Train the model
    svc_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = svc_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return svc_model, accuracy


def train_decision_tree_with_split(data, target_column, test_size=0.2, random_state=42, max_depth=None):
    """
    Train a DecisionTreeClassifier with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.
    max_depth (int or None): The maximum depth of the tree.

    Returns:
    model: Trained DecisionTreeClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the Decision Tree model
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_decision_tree(data, Tdata, target_column, random_state=42, max_depth=None):
    """
    Train a DecisionTreeClassifier with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - random_state (int): Seed for reproducibility.
    - max_depth (int or None): The maximum depth of the tree.

    Returns:
    - model: Trained DecisionTreeClassifier model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the Decision Tree model
    tree_model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)

    # Train the model
    tree_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = tree_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return tree_model, accuracy


def train_adaboost_with_split(data, target_column, test_size=0.2, random_state=42, n_estimators=50):
    """
    Train an AdaBoostClassifier with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.
    n_estimators (int): The maximum number of estimators at which boosting is terminated.

    Returns:
    model: Trained AdaBoostClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the AdaBoost model
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_adaboost(data, Tdata, target_column, random_state=42, n_estimators=50):
    """
    Train an AdaBoostClassifier with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - random_state (int): Seed for reproducibility.
    - n_estimators (int): The maximum number of estimators at which boosting is terminated.

    Returns:
    - model: Trained AdaBoostClassifier model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the AdaBoost model
    adaboost_model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model
    adaboost_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = adaboost_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return adaboost_model, accuracy


def train_gradient_boosting_with_split(data, target_column, test_size=0.2, random_state=42, n_estimators=100):
    """
    Train a GradientBoostingClassifier with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.
    n_estimators (int): The number of boosting stages to be run.

    Returns:
    model: Trained GradientBoostingClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the Gradient Boosting model
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_gradient_boosting(data, Tdata, target_column, random_state=42, n_estimators=100):
    """
    Train a GradientBoostingClassifier with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - random_state (int): Seed for reproducibility.
    - n_estimators (int): The number of boosting stages to be run.

    Returns:
    - model: Trained GradientBoostingClassifier model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the Gradient Boosting model
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model
    gb_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = gb_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return gb_model, accuracy


def train_neural_network_with_split(data, target_column, test_size=0.2, random_state=42, hidden_layer_sizes=(100,)):
    """
    Train a Multi-layer Perceptron (Neural Network) with an internal train-test split.

    Parameters:
    data (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the data to use for testing.
    random_state (int): Seed for reproducibility.
    hidden_layer_sizes (tuple): The number of neurons in each hidden layer.

    Returns:
    model: Trained MLPClassifier model.
    accuracy: Accuracy on the test set.
    """
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the neural network model
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy on the test set: {accuracy:.2f}")
    
    return model, accuracy


def train_and_evaluate_neural_network(data, Tdata, target_column, random_state=42, hidden_layer_sizes=(100,)):
    """
    Train a Multi-layer Perceptron (Neural Network) with the training dataset and evaluate it with the test dataset.

    Parameters:
    - data (DataFrame): Training dataset including the target column.
    - Tdata (DataFrame): Test dataset including the target column.
    - target_column (str): Name of the target column. Default is 'survived'.
    - random_state (int): Seed for reproducibility.
    - hidden_layer_sizes (tuple): The number of neurons in each hidden layer.

    Returns:
    - model: Trained MLPClassifier model.
    - accuracy: Accuracy of the model on the test dataset.
    """
    # Separate features and target in the training dataset
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Initialize the Neural Network model (MLPClassifier)
    nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)

    # Train the model
    nn_model.fit(X_train, y_train)

    # Separate features and target in the test dataset
    X_test = Tdata.drop(target_column, axis=1, errors='ignore')
    y_test = Tdata[target_column]

    # Make predictions on the test dataset
    y_pred = nn_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    print('Classification report:')
    print(classification_report(y_test, y_pred))

    return nn_model, accuracy


def evaluate_model_visuals(model, X_test, y_test, X_train, y_train):
    """
    Genera la matriz de confusión, la curva de aprendizaje y la curva de validación para un modelo entrenado.

    Parameters:
    - model: El modelo entrenado.
    - X_test (DataFrame o array): Conjunto de características de prueba.
    - y_test (Series o array): Conjunto de etiquetas de prueba.
    - X_train (DataFrame o array): Conjunto de características de entrenamiento.
    - y_train (Series o array): Conjunto de etiquetas de entrenamiento.
    """
    # Matriz de confusión
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.show()
    
    # Curva de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

    # Curva de validación para el hiperparámetro `max_depth` del RandomForest
    param_range = np.arange(1, 20, 2)
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name="max_depth", param_range=param_range, cv=5
    )
    plt.figure()
    plt.plot(param_range, np.mean(train_scores, axis=1), label="Training Score")
    plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation Score")
    plt.title("Validation Curve for max_depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


def train_random_forest_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un RandomForestClassifier con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo RandomForestClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_random_forest_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo RandomForestClassifier usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo RandomForestClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def train_knn_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un KNeighborsClassifier con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo KNeighborsClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_knn_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo KNeighborsClassifier usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo KNeighborsClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def train_linear_svc_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un modelo LinearSVC con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo LinearSVC entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],          # Parámetro de regularización
        'max_iter': [1000, 5000, 10000],  # Máximo número de iteraciones
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=LinearSVC(random_state=random_state, dual=False),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_linear_svc_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo LinearSVC usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo LinearSVC entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],          # Parámetro de regularización
        'max_iter': [1000, 5000, 10000],  # Máximo número de iteraciones
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=LinearSVC(random_state=random_state, dual=False),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def train_decision_tree_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un modelo DecisionTreeClassifier con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo DecisionTreeClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_decision_tree_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo DecisionTreeClassifier usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo DecisionTreeClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def train_adaboost_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un modelo AdaBoostClassifier con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo AdaBoostClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de estimadores en AdaBoost
        'learning_rate': [0.01, 0.1, 1],  # Tasa de aprendizaje
        'base_estimator': [None, DecisionTreeClassifier(max_depth=1)]  # Estimador base (DecisionTreeClassifier de profundidad 1)
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=AdaBoostClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_adaboost_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo AdaBoostClassifier usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo AdaBoostClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de estimadores en AdaBoost
        'learning_rate': [0.01, 0.1, 1],  # Tasa de aprendizaje
        'base_estimator': [None, DecisionTreeClassifier(max_depth=1)]  # Estimador base (DecisionTreeClassifier de profundidad 1)
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=AdaBoostClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def train_gradient_boosting_with_grid_search(data, target_column, test_size=0.2, random_state=42):
    """
    Entrena un modelo GradientBoostingClassifier con una división interna de entrenamiento-prueba y GridSearchCV para optimizar hiperparámetros.

    Parameters:
    data (DataFrame): El DataFrame de entrada.
    target_column (str): El nombre de la columna objetivo.
    test_size (float): Proporción de datos para el conjunto de prueba.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: El modelo GradientBoostingClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de árboles en el modelo
        'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
        'max_depth': [3, 5, 7],  # Profundidad máxima de los árboles
        'subsample': [0.8, 1.0],  # Fracción de muestras a usar en cada estimador
        'min_samples_split': [2, 5],  # Mínimo de muestras para dividir un nodo
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)

    return best_model, best_accuracy


def train_and_evaluate_gradient_boosting_with_grid_search(data, Tdata, target_column, random_state=42):
    """
    Entrena un modelo GradientBoostingClassifier usando GridSearchCV para optimizar hiperparámetros
    con el conjunto de datos de entrenamiento y evalúa el modelo con el conjunto de prueba.

    Parameters:
    data (DataFrame): El conjunto de datos de entrenamiento que incluye la columna objetivo.
    Tdata (DataFrame): El conjunto de datos de prueba que incluye la columna objetivo.
    target_column (str): Nombre de la columna objetivo.
    random_state (int): Semilla para reproducibilidad.

    Returns:
    best_model: Modelo GradientBoostingClassifier entrenado con los mejores hiperparámetros.
    best_accuracy: Precisión del modelo en el conjunto de prueba.
    """
    # Separar características y objetivo en el conjunto de entrenamiento
    X_train = data.drop(target_column, axis=1)
    y_train = data[target_column]

    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de estimadores en el modelo
        'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
        'max_depth': [3, 5, 7],  # Profundidad máxima de los árboles
        'subsample': [0.8, 1.0],  # Fracción de muestras a usar en cada estimador
        'min_samples_split': [2, 5],  # Mínimo de muestras para dividir un nodo
    }

    # Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo y hacer predicciones en el conjunto de prueba
    best_model = grid_search.best_estimator_
    X_test = Tdata.drop(target_column, axis=1)
    y_test = Tdata[target_column]
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    print(f"Best accuracy on the test set: {best_accuracy:.2f}")
    print("Best parameters:", grid_search.best_params_)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return best_model, best_accuracy


def missing_data_summary(data):
    # Contar los valores faltantes "?" en cada columna
    missing_counts = data.apply(lambda x: (x == '?').sum())
    
    # Crear una tabla con los resultados
    missing_data_table = pd.DataFrame({'Feature': missing_counts.index, 'Missing Values': missing_counts.values})
    
    # Filtrar para mostrar solo las columnas con valores faltantes
    missing_data_table = missing_data_table[missing_data_table['Missing Values'] > 0].reset_index(drop=True)
    
    return missing_data_table


def apply_kmeans(X, n_clusters=6, random_state=42):
    """
    Aplica el algoritmo K-Means para agrupar los datos en n_clusters.

    Parameters:
    - X (ndarray o DataFrame): Datos de entrada (características).
    - n_clusters (int): Número de clusters a formar.
    - random_state (int): Semilla para reproducibilidad.

    Returns:
    - kmeans (KMeans): El modelo de KMeans entrenado.
    - labels (ndarray): Etiquetas de los clusters asignadas a cada punto.
    - inertia (float): Inercia (suma de las distancias al centro del cluster más cercano).
    - X_with_labels (ndarray): Datos con las etiquetas de los clusters añadidas como una nueva columna.
    """
    # Crear y entrenar el modelo de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    
    # Obtener las etiquetas y la inercia
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    
    print(f"Inercia: {inertia}")
    
    # Añadir la etiqueta de KMeans como una nueva columna
    X_with_labels = np.column_stack((X, labels))
    
    # Visualizar los resultados (si son 2D)
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
        plt.title('K-Means Clustering')
        plt.show()

    return kmeans, labels, inertia, X_with_labels


def train_logistic_regression_with_kmeans(X, y, test_size=0.2, random_state=42):
    """
    Entrena un modelo de regresión logística utilizando las características generadas por KMeans.
    
    Parameters:
    - X (ndarray): El conjunto de datos de características.
    - y (ndarray): Las etiquetas del conjunto de datos.
    - test_size (float): Proporción de datos para el conjunto de prueba.
    - random_state (int): Semilla para la reproducibilidad.
    
    Returns:
    - logreg: El modelo de regresión logística entrenado.
    - accuracy: Precisión del modelo sobre el conjunto de prueba.
    """
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Entrenar un modelo de regresión logística
    logreg = LogisticRegression(random_state=random_state)
    logreg.fit(X_train, y_train)
    
    # Hacer predicciones sobre el conjunto de prueba
    y_pred = logreg.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Logistic Regression with KMeans labels: {accuracy:.2f}")
    
    # Visualización de las predicciones
    plt.figure(figsize=(8, 6))
    
    # Crear una malla de puntos
    h = .02  # Step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predecir sobre la malla de puntos
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Dibujar el gráfico
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolors='k', marker='o', cmap='viridis', alpha=0.6)
    plt.title('Logistic Regression - Predicciones sobre datos con KMeans')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    return logreg, accuracy


def train_logistic_regression_with_pca(X, y, test_size=0.2, random_state=42):
    """
    Entrena un modelo de regresión logística utilizando las características generadas por KMeans.
    
    Parameters:
    - X (ndarray): El conjunto de datos de características.
    - y (ndarray): Las etiquetas del conjunto de datos.
    - test_size (float): Proporción de datos para el conjunto de prueba.
    - random_state (int): Semilla para la reproducibilidad.
    
    Returns:
    - logreg: El modelo de regresión logística entrenado.
    - accuracy: Precisión del modelo sobre el conjunto de prueba.
    """
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Entrenar un modelo de regresión logística
    logreg = LogisticRegression(random_state=random_state)
    logreg.fit(X_train, y_train)
    
    # Hacer predicciones sobre el conjunto de prueba
    y_pred = logreg.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Logistic Regression with PCA features: {accuracy:.2f}")
    
    # Visualización de las predicciones
    plt.figure(figsize=(8, 6))
    
    # Crear una malla de puntos
    h = .02  # Step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predecir sobre la malla de puntos con 2 características (sin la columna extra de ceros)
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Dibujar el gráfico
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolors='k', marker='o', cmap='viridis', alpha=0.6)
    plt.title('Logistic Regression - Predicciones sobre datos con PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()
    
    return logreg, accuracy


def apply_pca(X, n_components=2):
    """
    Aplica PCA para reducir la dimensionalidad de los datos.

    Parameters:
    - X (DataFrame or ndarray): Datos de entrada (características).
    - n_components (int): Número de componentes principales para mantener.

    Returns:
    - pca: Modelo PCA entrenado.
    - X_pca: Datos proyectados a las componentes principales.
    """
    # Crear y entrenar el modelo PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
    
    # Visualizar los resultados en 2D si n_components=2
    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue')
        plt.title('PCA - Proyección a 2D')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.show()

    return pca, X_pca
