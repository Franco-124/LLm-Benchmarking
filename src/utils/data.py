from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

@dataclass
class DiabetesDataset:
    df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_std: np.ndarray
    X_test_std: np.ndarray

def get_diabetes_data(cedula_terminacion: int = 72) -> DiabetesDataset:
    """
    Loads the diabetes dataset and prepares it for modeling.
    
    Args:
        cedula_terminacion (int): The last two digits of the user's ID to determine the split.

    Returns:
        DiabetesDataset: Data class containing the dataset's splits and original reference df.
    """
    diabetes = load_diabetes()

    X = diabetes.data
    y = diabetes.target

    df = pd.DataFrame(X, columns=diabetes.feature_names)
    df["target"] = y

    df = df.drop(['age', 'sex', 's1', 's2', 's3', 's4', 's6'], axis=1)
    df = df.rename(columns={
        'bmi': 'IMC',
        'bp': 'Presion',
        's5': 'Trigliceridos_log',
    })

    X_filtered = df[['IMC', 'Presion', 'Trigliceridos_log']].values
    y_filtered = df['target'].values

    porcentaje = cedula_terminacion / 100.0
    if cedula_terminacion >= 50:
        train_size = porcentaje
        test_size = 1.0 - train_size
    else:
        test_size = porcentaje
        train_size = 1.0 - test_size
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=test_size, random_state=42
    )

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_std = (X_train - X_mean) / X_std
    X_test_std  = (X_test  - X_mean) / X_std

    return DiabetesDataset(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_std=X_train_std,
        X_test_std=X_test_std
    )
