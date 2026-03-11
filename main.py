import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

from src.data import get_diabetes_data
from src.models.genetic import run_genetic_algorithm
from src.models.ensemble import run_bagging_ensemble
from src.models.neural_net import run_neural_networks_benchmarking
from src.utils import plot_comparisons, plot_fitness_evolution
import uvicorn
# Configuración de Logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_execution")

app = FastAPI(
    title="ML Models Benchmark API",
    description="API para orquestar los modelos de predicción de Diabetes",
    version="1.0.0"
)

class ModelExecutionError(Exception):
    """Excepción custom para errores del dominio en la ejecución de los modelos."""
    pass

@app.get("/execute", response_model=Dict[str, Any])
def execute_models() -> Dict[str, Any]:
    """
    Ejecuta el pipeline de datos, evalúa los tres modelos con validación cruzada.
    Retorna métricas de evaluación (MSE, RMSE, MAE, R2) en estructura JSON.
    """
    try:
        logger.info("Cargando dataset y aplicando preprocesamiento...")
        dataset = get_diabetes_data(cedula_terminacion=72)
        
        # Preparando matrices enteras para K-Fold
        X_full = dataset.df[['IMC', 'Presion', 'Trigliceridos_log']].values
        y_full = dataset.df['target'].values
        
        # Estandarizacion para Keras
        X_mean = X_full.mean(axis=0, keepdims=True)
        X_std = X_full.std(axis=0, keepdims=True) + 1e-8
        X_scaled = (X_full - X_mean) / X_std

        logger.info("Ejecutando algoritmo genético optimizado (K-Fold 5)...")
        ga_metrics, ga_params, fitness_history = run_genetic_algorithm(X_full, y_full, k_folds=5)

        logger.info("Ejecutando ensamble bagging regressor (K-Fold 5)...")
        bagging_metrics, bagging_model = run_bagging_ensemble(X_full, y_full, k_folds=5)

        logger.info("Ejecutando arquitecturas de Red Neuronal (K-Fold 5)...")
        nn_results, nn_models = run_neural_networks_benchmarking(X_scaled, y_full, k_folds=5)

        logger.info("Generando visualizaciones en carpeta local 'plots/'...")
        # 1. Graficar evolución del genético (último fold)
        plot_fitness_evolution(fitness_history)

        # 2. Graficar comparación (usamos arquitectura A como referencia de RNA)
        plot_comparisons(
            df=dataset.df,
            mejor_ga=ga_params,
            bagging_model=bagging_model,
            nn_model=nn_models["arquitectura_A"],
            X_mean=X_mean,
            X_std=X_std
        )

        logger.info("Los modelos finalizaron con exito. Retornando metricas.")

        return {
            "genetic_algorithm": {
                "mse": ga_metrics["mse"],
                "rmse": ga_metrics["rmse"],
                "r2": ga_metrics["r2"]
            },
            "bagging_ensemble": {
                "mse": round(bagging_metrics["mse"], 4),
                "rmse": round(bagging_metrics["rmse"], 4),
                "r2": round(bagging_metrics["r2"], 4)
            },
            "red_neuronal_keras": {
                k: {
                    "mse": round(v["mse"], 4),
                    "rmse": round(v["rmse"], 4),
                    "r2": round(v["r2"], 4)
                } for k, v in nn_results.items()
            }
        }

    except ValueError as ve:
        logger.error(f"Error procesando los valores en ejecución: {str(ve)}")
        raise HTTPException(status_code=422, detail="Error en procesamiento de datos o parámetros.")
    
    except Exception as exc:
        logger.error(f"Error critico comprobando el pipeline MLP: {str(exc)}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Fallo inesperado al correr la API.")

uvicorn.run(app, host="127.0.0.1", port=8000) if __name__ == "__main__" else None