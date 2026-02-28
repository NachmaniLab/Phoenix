import random
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from scripts.consts import ALL_CELLS, SEED, CELL_TYPE_COL


def get_train_target(
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
    ) -> pd.Series:
    if scaled_pseudotime is not None:
        return scaled_pseudotime.loc[:, lineage].dropna()
    
    if cell_type == ALL_CELLS:
        return cell_types[CELL_TYPE_COL]  # type: ignore[index]
    return cell_types[CELL_TYPE_COL] == cell_type  # type: ignore[index]


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.astype("category").cat.codes.to_numpy()


def get_train_data(
        scaled_expression: pd.DataFrame,
        features: list[str] | None = None,
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
        set_size: int | None = None,
        feature_selection: str | None = None,
        seed: int = SEED
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    feature_selection: either 'ANOVA' or 'RF', supported for both classification and regression
    """
    assert (cell_types is not None and cell_type is not None) or (scaled_pseudotime is not None and lineage is not None)
    is_regression = scaled_pseudotime is not None

    y = get_train_target(cell_types, scaled_pseudotime, cell_type, lineage)
    if features is None:
        features = scaled_expression.columns.tolist()
    else:
        cols = set(scaled_expression.columns)
        features = [f for f in features if f in cols]

    row_idx = scaled_expression.index.get_indexer(y.index)
    col_idx = scaled_expression.columns.get_indexer(features)
    X = scaled_expression.to_numpy(copy=False)[np.ix_(row_idx, col_idx)]

    set_size = min(set_size, len(features)) if set_size is not None else len(features)

    # Select best features using either ANOVA or RF
    if feature_selection is not None:
        if feature_selection == 'ANOVA':
            selected_features = SelectKBest(score_func=f_regression if is_regression else f_classif, k=set_size).fit(X, y)
            selected_indices = selected_features.get_support(indices=True)
            selected_genes = [features[i] for i in selected_indices]
            return selected_features.transform(X), y.to_numpy() if is_regression else encode_labels(y), selected_genes
        
        if feature_selection == 'RF':
            if is_regression:
                importances = RandomForestRegressor(random_state=seed, n_estimators=50).fit(X, y).feature_importances_
            else:
                importances = RandomForestClassifier(random_state=seed, class_weight='balanced', n_estimators=50).fit(X, y).feature_importances_
            selected_indices = (-importances).argsort()[:set_size]
            selected_genes = [features[i] for i in selected_indices]
            return X[:, selected_indices], y.to_numpy() if is_regression else encode_labels(y), selected_genes
        
        raise ValueError(f'Unsupported feature selection method {feature_selection}')

    # Select randomly
    selected_indices = random.Random(seed).sample(list(range(X.shape[1])), set_size)
    return X[:, selected_indices], y.to_numpy() if is_regression else encode_labels(y), [features[i] for i in selected_indices]


def train(X: np.ndarray, y: np.ndarray, model, score_function, cv) -> tuple[float, np.ndarray]:
    scores = []
    importances = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        fold_model = clone(model)  # ensure fresh model each fold
        fold_model.fit(X_train, y_train)
        scores.append(score_function(fold_model, X_val, y_val))
        importances.append(fold_model.feature_importances_)
    return float(np.median(scores)), np.median(np.vstack(importances), axis=0)


def create_cv(is_regression: bool, n_splits: int):
    if is_regression:
        return KFold(n_splits=n_splits, shuffle=False)
    return StratifiedKFold(n_splits=n_splits, shuffle=False)


def get_prediction_score(
        scaled_expression: pd.DataFrame,
        predictor,
        predictor_args: dict,
        score_function,
        cv,
        seed: int,
        gene_set: list[str] | None = None,
        set_size: int | None = None,
        feature_selection: str | None = None,
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
    ) -> tuple[float, list[str], list[float]]:
    X, y, selected_genes = get_train_data(
        scaled_expression=scaled_expression,
        features=gene_set,
        cell_types=cell_types,
        scaled_pseudotime=scaled_pseudotime,
        cell_type=cell_type,
        lineage=lineage,
        set_size=set_size,
        feature_selection=feature_selection,
        seed=seed,
    )
    score, gene_importances = train(
        X=X,
        y=y,
        model=predictor(**dict(predictor_args)),
        score_function=score_function,
        cv=cv,
    )
    importance_series = pd.Series(
        gene_importances, index=selected_genes, name='median_importance'
    ).sort_values(ascending=False)
    return score, importance_series.index.tolist(), importance_series.tolist()


def compare_scores(pathway_score: float, background_scores: list[float], distribution: str) -> float:

    if all([s == pathway_score for s in background_scores]):
        p_value = np.NaN

    elif distribution == 'normal':
        alternative = 'less'  # background is less than pathway
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Precision loss occurred in moment calculation')
            p_value = stats.ttest_1samp(background_scores, pathway_score, alternative=alternative)[1]

    elif distribution == 'gamma':
        try:
            shape, loc, scale = stats.gamma.fit(background_scores)
            cdf_value = stats.gamma.cdf(pathway_score, shape, loc, scale)
            p_value = 1 - cdf_value
        except stats._warnings_errors.FitError:
            p_value = np.NaN
        
    else:
        raise ValueError('Unsupported distribution type. Use `normal` or `gamma`')

    return p_value if not np.isnan(p_value) else 1.0
