import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


def run_bagging_ensemble(X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> tuple[dict[str, float], BaggingRegressor]:
    """
    Trains and evaluates a BaggingRegressor with a DecisionTreeRegressor base estimator using K-Fold CV.

    Args:
        X: Full features array.
        y: Full target array.
        k_folds: Number of folds for cross validation.

    Returns:
        Dict[str, float]: Average metrics (mse, rmse, mae, r2) across all folds.
    """
    base_tree = DecisionTreeRegressor(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    bagging = BaggingRegressor(
        estimator=base_tree,
        n_estimators=300,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=42
    )

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]

        bagging.fit(X_train_kf, y_train_kf)
        pred_test = bagging.predict(X_test_kf)
        
        mse = mean_squared_error(y_test_kf, pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_kf, pred_test)
        r2 = r2_score(y_test_kf, pred_test)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return {
        "mse": float(np.mean(mse_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "mae": float(np.mean(mae_scores)),
        "r2": float(np.mean(r2_scores))
    }, bagging
