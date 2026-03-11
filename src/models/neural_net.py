import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from typing import Dict, Any

def build_architecture_a(input_dim: int) -> Sequential:
    """
    Architecture A (Simple):
    - 1 Hidden Layer (16 neurons, ReLU)
    - Optimizer: Adam (lr=0.01)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

def build_architecture_b(input_dim: int) -> Sequential:
    """
    Architecture B (Deep & Narrow):
    - 3 Hidden Layers (8, 8, 8 neurons, ReLU)
    - Optimizer: RMSprop (lr=0.001)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')
    return model

def build_architecture_c(input_dim: int) -> Sequential:
    """
    Architecture C (Wide & Dropout):
    - 2 Hidden Layers (64, 32 neurons, ReLU) + Dropout
    - Optimizer: Adam (lr=0.005)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    return model

def run_neural_networks_benchmarking(X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Sequential]]:
    """
    Trains and evaluates 3 different Keras MLP architectures using K-Fold Cross Validation.

    Args:
        X: Standardized features array.
        y: Target array.
        k_folds: Number of folds for cross validation.

    Returns:
        Dict[str, Dict[str, float]]: dictionary holding mse, rmse, mae, r2 metrics for each architecture.
    """
    
    tf.random.set_seed(42)
    input_dim = X.shape[1]
    
    # K-Fold definitor
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store all architectures results
    results: Dict[str, Dict[str, float]] = {
        "arquitectura_A": {"mse": [], "rmse": [], "mae": [], "r2": []},
        "arquitectura_B": {"mse": [], "rmse": [], "mae": [], "r2": []},
        "arquitectura_C": {"mse": [], "rmse": [], "mae": [], "r2": []}
    }
    
    # Fold Iteration
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]
        
        # We must rebuild models every fold so weights reset
        architectures = {
            "arquitectura_A": build_architecture_a(input_dim),
            "arquitectura_B": build_architecture_b(input_dim),
            "arquitectura_C": build_architecture_c(input_dim)
        }
        
        for name, model in architectures.items():
            model.fit(X_train_kf, y_train_kf, epochs=100, batch_size=32, verbose=0)
            
            y_pred = model.predict(X_test_kf, verbose=0).ravel()
            
            mse = mean_squared_error(y_test_kf, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_kf, y_pred)
            r2 = r2_score(y_test_kf, y_pred)
            
            results[name]["mse"].append(mse)
            results[name]["rmse"].append(rmse)
            results[name]["mae"].append(mae)
            results[name]["r2"].append(r2)
            
        fold += 1
        
    # Average the metrics
    final_results = {}
    for name, metrics in results.items():
        final_results[name] = {
            "mse": float(np.mean(metrics["mse"])),
            "rmse": float(np.mean(metrics["rmse"])),
            "mae": float(np.mean(metrics["mae"])),
            "r2": float(np.mean(metrics["r2"]))
        }
        
    return final_results, architectures
