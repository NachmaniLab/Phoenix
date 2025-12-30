import random, inspect
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from scripts.consts import ALL_CELLS, SEED, CELL_TYPE_COL, METRICS
from scripts.utils import show_runtime


def get_train_target(
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
    ):
    if scaled_pseudotime is not None:
        return scaled_pseudotime.loc[:, lineage].dropna()
    
    if cell_type == ALL_CELLS:
        return cell_types[CELL_TYPE_COL]  # type: ignore[index]
    return cell_types[CELL_TYPE_COL] == cell_type  # type: ignore[index]


def get_train_data(
        scaled_expression: pd.DataFrame,
        features: list[str] | None = None,
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
        set_size: int | None = None,
        feature_selection: str | None = None,
        selection_args: dict = {},
        ordered_selection: bool = False,
        seed: int = SEED
    ) -> tuple[np.ndarray, pd.Series, list[str], list[float] | None]:
    """
    feature_selection: either 'ANOVA' or 'RF', supported for both classification and regression
    ordered_selection: ignored if feature_selection is set
    """
    assert (cell_types is not None and cell_type is not None) or (scaled_pseudotime is not None and lineage is not None)
    is_regression = scaled_pseudotime is not None

    y = get_train_target(cell_types, scaled_pseudotime, cell_type, lineage)
    cells = y.index
    features = [f for f in features if f in scaled_expression.columns] if features is not None else scaled_expression.columns
    X = scaled_expression.loc[cells, features].to_numpy()

    set_size = min(set_size, len(features)) if set_size is not None else len(features)

    # Select best features using either ANOVA or RF
    if feature_selection is not None:
        if feature_selection == 'ANOVA':
            selected_features = SelectKBest(score_func=f_regression if is_regression else f_classif, k=set_size).fit(X, y)
            selected_indices = selected_features.get_support(indices=True)
            selected_genes = [features[i] for i in selected_indices]
            importances = selected_features.scores_[selected_indices].tolist()
            return selected_features.transform(X), y, selected_genes, importances
        
        if feature_selection == 'RF':
            if 'n_estimators' not in selection_args.keys():
                selection_args['n_estimators'] = 50
            if is_regression:
                importances = RandomForestRegressor(random_state=seed, **selection_args).fit(X, y).feature_importances_
            else:
                importances = RandomForestClassifier(random_state=seed, class_weight='balanced', **selection_args).fit(X, y).feature_importances_
            selected_indices = (-importances).argsort()[:set_size]
            selected_genes = [features[i] for i in selected_indices]
            return X[:, selected_indices], y, selected_genes, importances[selected_indices].tolist()
        
        raise ValueError(f'Unsupported feature selection method {feature_selection}')

    # Select first
    if ordered_selection:
        return X[:, :set_size], y, features[:set_size], None

    # Select randomly
    selected_indices = random.Random(seed).sample(list(range(X.shape[1])), set_size)
    return X[:, selected_indices], y, [features[i] for i in selected_indices], None


@show_runtime
def train(
        X, y,
        predictor,
        predictor_args: dict,
        metric: str,
        cross_validation: int | None = None,
        balanced_weights: bool = True,
        train_size: float = 0.8,
        bins: int = 3,
        seed: int = SEED,
    ) -> float:

    if 'n_jobs' in inspect.signature(predictor).parameters:
        predictor_args['n_jobs'] = -1  # all processes
    if 'random_state' in inspect.signature(predictor).parameters:
        predictor_args['random_state'] = seed
    if balanced_weights and 'class_weight' in inspect.signature(predictor).parameters:
        predictor_args['class_weight'] = 'balanced'

    model = predictor(**predictor_args)
    score_func = make_scorer(METRICS[metric], greater_is_better=True)

    if isinstance(y.iloc[0], str):
        y = LabelEncoder().fit_transform(y)

    if cross_validation:
        score = np.median(cross_val_score(model, X, y, cv=cross_validation, scoring=score_func))

    else:
        stratify = pd.cut(y, bins=bins, labels=False) if y.dtype == float else y
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, train_size=train_size, random_state=seed)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = score_func(y_test, y_pred)

    del model
    return float(score)
