import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


def fit_ga(X_train: np.ndarray, y_train: np.ndarray, individuos: int = 200, max_generaciones: int = 200) -> tuple[dict[str, float], list[float]]:
    X1_np = X_train[:, 0]
    X2_np = X_train[:, 1]
    X3_np = X_train[:, 2]
    Y_np = y_train

    np.random.seed(42)
    random.seed(42)

    poblacion = pd.DataFrame({
        'm1': np.random.uniform(-30,30,individuos),
        'm2': np.random.uniform(-30,30,individuos),
        'm3': np.random.uniform(-30,30,individuos),
        'b':  np.random.uniform(-30,30,individuos),
        'fitness': 0.0
    })

    # Fitness inicial vectorizado en Numpy
    m1 = poblacion["m1"].values
    m2 = poblacion["m2"].values
    m3 = poblacion["m3"].values
    b = poblacion["b"].values
    
    y_pred = (m1[:, None] * X1_np + 
              m2[:, None] * X2_np + 
              m3[:, None] * X3_np + 
              b[:, None])
              
    mse = np.mean((Y_np - y_pred)**2, axis=1)
    poblacion["fitness"] = mse**0.5

    cantidad = poblacion.sort_values('fitness', ascending=True).reset_index(drop=True)
    fitness_history = [float(cantidad.iloc[0]['fitness'])]
    generacion = 0

    while cantidad.iloc[0]['fitness'] > 50.0 and generacion < max_generaciones:
        generacion += 1
        nuevos = []

        for _ in range(individuos):
            padres = cantidad.sample(2, replace=False).reset_index(drop=True)
            p1, p2 = padres.loc[0], padres.loc[1]

            alpha = random.uniform(0.3, 0.7)
            hijo = {h: alpha*p1[h] + (1-alpha)*p2[h] for h in ['m1','m2','m3','b']}

            ratio = 0.25
            for h in ['m1','m2','m3','b']:
                if random.random() < ratio:
                    hijo[h] += random.gauss(0, 50.0)
                    lim = 2000.0 if h != 'b' else 500.0
                    hijo[h] = max(-lim, min(lim, hijo[h]))

            hijo['fitness'] = 0.0
            nuevos.append(hijo)

        todo = pd.concat([cantidad, pd.DataFrame(nuevos)], ignore_index=True)

        m1 = todo["m1"].values
        m2 = todo["m2"].values
        m3 = todo["m3"].values
        b = todo["b"].values
        
        y_pred = (m1[:, None] * X1_np + 
                  m2[:, None] * X2_np + 
                  m3[:, None] * X3_np + 
                  b[:, None])
                  
        mse = np.mean((Y_np - y_pred)**2, axis=1)
        todo["fitness"] = mse**0.5

        cantidad = todo.sort_values('fitness', ascending=True).head(individuos).reset_index(drop=True)
        fitness_history.append(float(cantidad.iloc[0]['fitness']))

    params = {
        'm1': cantidad.iloc[0]['m1'],
        'm2': cantidad.iloc[0]['m2'],
        'm3': cantidad.iloc[0]['m3'],
        'b':  cantidad.iloc[0]['b']
    }
    return params, fitness_history


def run_genetic_algorithm(X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> tuple[dict[str, float], dict[str, float], list[float]]:
    """
    Entrena y evalúa las predicciones generadas por los parámetros de un modelo Genético.
    Realiza validación por K-Fold y vectorización de métricas.

    Args:
        X: Características (IMC, Presión, Triglicéridos).
        y: Target original.
        k_folds: Total iteraciones de K-Fold CV.

    Returns:
        Dict: Métricas calculadas.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]

        best_params, fitness_history = fit_ga(X_train_kf, y_train_kf, individuos=200, max_generaciones=50)
        
        y_pred = (best_params['m1'] * X_test_kf[:, 0] + 
                  best_params['m2'] * X_test_kf[:, 1] + 
                  best_params['m3'] * X_test_kf[:, 2] + 
                  best_params['b'])
                  
        mse = mean_squared_error(y_test_kf, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_kf, y_pred)
        r2 = r2_score(y_test_kf, y_pred)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return {
        "mse": float(np.mean(mse_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "mae": float(np.mean(mae_scores)),
        "r2": float(np.mean(r2_scores))
    }, best_params, fitness_history
