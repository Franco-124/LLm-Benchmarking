import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict

def plot_fitness_evolution(mejores: List[float], output_dir: str = "plots") -> None:
    """
    Plots the evolution of the best fitness per generation and saves it to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mejores)+1), mejores, marker='o')
    plt.title('Evolución del fitness por generación')
    plt.xlabel('Generación')
    plt.ylabel('Mejor fitness')
    plt.grid(True)
    
    save_path = os.path.join(output_dir, 'evolucion_fitness.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparisons(df: pd.DataFrame, 
                     mejor_ga: Dict[str, float], 
                     bagging_model, 
                     nn_model, 
                     X_mean: np.ndarray, 
                     X_std: np.ndarray,
                     output_dir: str = "plots") -> None:
    """
    Plots the comparison between the three models against the real data and saves it.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_plot = df['IMC'].values
    Y_plot = df['target'].values
    
    x_range = np.linspace(X_plot.min(), X_plot.max(), 200)
    presion_mean = float(df['Presion'].mean())
    tri_mean     = float(df['Trigliceridos_log'].mean())

    m1_ga, m2_ga, m3_ga, b_ga = mejor_ga['m1'], mejor_ga['m2'], mejor_ga['m3'], mejor_ga['b']
    
    y_ga = (
        m1_ga * x_range +
        m2_ga * presion_mean +
        m3_ga * tri_mean +
        b_ga
    )
    y_ga = np.asarray(y_ga).ravel()

    X_bag = np.column_stack([
        x_range,
        np.full_like(x_range, presion_mean),
        np.full_like(x_range, tri_mean)
    ])
    y_bag = bagging_model.predict(X_bag)
    y_bag = np.asarray(y_bag).ravel()

    X_mean_flat = np.asarray(X_mean).ravel()
    X_std_flat  = np.asarray(X_std).ravel()

    x_nn_scaled    = (x_range      - X_mean_flat[0]) / X_std_flat[0]
    presion_scaled = (presion_mean  - X_mean_flat[1]) / X_std_flat[1]
    tri_scaled     = (tri_mean      - X_mean_flat[2]) / X_std_flat[2]
    
    X_nn = np.column_stack([
        x_nn_scaled,
        np.full_like(x_range, presion_scaled),
        np.full_like(x_range, tri_scaled)
    ])
    
    y_nn = nn_model.predict(X_nn, verbose=0).ravel()

    plt.figure(figsize=(10,6))
    plt.scatter(X_plot, Y_plot, color='gray', alpha=0.4, label='Datos Reales')

    plt.plot(x_range, y_ga, '--', label='Algoritmo Genético', color='green', linewidth=2)
    plt.plot(x_range, y_bag, '-.', label='Ensamblaje (Bagging)', color='red', linewidth=2)
    plt.plot(x_range, y_nn, '-', label='Red Neuronal (Keras)', color='blue', linewidth=2)

    plt.title('Comparación de Modelos Predictores (Progresión vs IMC)')
    plt.xlabel('IMC')
    plt.ylabel('Progresión Enfermedad (Target)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    save_path = os.path.join(output_dir, 'comparacion_modelos.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    