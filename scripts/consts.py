import numpy as np
from enum import Enum, auto
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from scripts.metrics import weighted_metric_using_icf, compute_f1, compute_recall


class BackgroundMode(Enum):
    REAL = auto()
    RANDOM = auto()
    AUTO = auto()


SIZES = [2, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
LEN_SIZES = len(SIZES)

SEED = 3407

REDUCTION_METHODS = ['pca', 'umap', 'tsne']

DATABASES = ['go', 'kegg', 'msigdb']

CLASSIFIERS = {
    'Reg': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'SVM': SVC,
    'DTree': DecisionTreeClassifier,
    'RF': RandomForestClassifier,
    'LGBM': LGBMClassifier,
    'XGB': XGBClassifier,
    'GradBoost': GradientBoostingClassifier,
    'MLP': MLPClassifier
}
REGRESSORS = {
    'Reg': LinearRegression,
    'KNN': KNeighborsRegressor,
    'SVM': SVR,
    'DTree': DecisionTreeRegressor,
    'RF': RandomForestRegressor,
    'LGBM': LGBMRegressor,
    'XGB': XGBRegressor,
    'GradBoost': GradientBoostingRegressor,
    'MLP': MLPRegressor
}
assert CLASSIFIERS.keys() == REGRESSORS.keys()

CLASSIFIER_ARGS = {
    LogisticRegression: {'max_iter': 300, 'n_jobs': -1, 'class_weight': 'balanced'},
    KNeighborsClassifier: {'n_neighbors': 10, 'n_jobs': -1},
    SVC: {'kernel': 'rbf', 'class_weight': 'balanced'},
    DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 10, 'class_weight': 'balanced', 'random_state': SEED},
    RandomForestClassifier: {'criterion': 'entropy', 'max_depth': 20, 'n_estimators': 20, 'n_jobs': -1, 'class_weight': 'balanced', 'random_state': SEED},
    LGBMClassifier: {'n_estimators': 20, 'verbose': -1, 'n_jobs': -1, 'class_weight': 'balanced'},
    XGBClassifier: {'n_estimators': 20},
    GradientBoostingClassifier: {'n_estimators': 20},
    MLPClassifier: {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 5000},
}
REGRESSOR_ARGS = {
    LinearRegression: {'fit_intercept': True, 'n_jobs': -1},
    KNeighborsRegressor: {'n_neighbors': 10, 'n_jobs': -1},
    SVR: {'kernel': 'rbf'},
    DecisionTreeRegressor: {'max_depth': 10, 'random_state': SEED},
    RandomForestRegressor: {'criterion': 'squared_error', 'max_depth': 20, 'n_estimators': 20, 'n_jobs': -1, 'random_state': SEED},
    LGBMRegressor: {'n_estimators': 20, 'verbose': -1, 'n_jobs': -1},
    XGBRegressor: {'n_estimators': 20},
    GradientBoostingRegressor: {'n_estimators': 20},
    MLPRegressor: {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 5000},
}

CLASSIFICATION_METRICS = {
    'accuracy': accuracy_score,
    'accuracy_balanced': balanced_accuracy_score,
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary'),
    'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
    'f1_weighted_icf': lambda y_true, y_pred: weighted_metric_using_icf(y_true, y_pred, compute_f1),
    'recall_weighted_icf': lambda y_true, y_pred: weighted_metric_using_icf(y_true, y_pred, compute_recall),
}
REGRESSION_METRICS = {
    'neg_mean_absolute_error': lambda y_true, y_pred: -1 * mean_absolute_error(y_true, y_pred),
    'neg_mean_squared_error': lambda y_true, y_pred: -1 * mean_squared_error(y_true, y_pred),
    'neg_root_mean_squared_error': lambda y_true, y_pred: -1 * np.sqrt(mean_squared_error(y_true, y_pred))
}
METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}

FEATURE_SELECTION_METHODS = ['ANOVA', 'RF']  # TODO: add `random` instead of using None
DISTRIBUTIONS = ['gamma', 'normal']

ALL_CELLS = 'All'
OTHER_CELLS = 'Other'
TARGET_COL = 'target'
CELL_TYPE_COL = 'cell_type'

LIST_SEP = '; '

# Defaults
NUM_GENES = 5000  # TODO: add param
REDUCTION = 'umap'
DB = 'ALL'
CLASSIFIER = 'RF'
REGRESSOR = 'RF'
CLASSIFICATION_METRIC = 'f1_weighted_icf'
REGRESSION_METRIC = 'neg_root_mean_squared_error'
CROSS_VALIDATION = 10
REPEATS = 150
FEATURE_SELECTION = 'RF'
SET_FRACTION = 0.75
MIN_SET_SIZE = SIZES[0]
THRESHOLD = 0.05  # TODO: add param
EFFECT_SIZE_THRESHOLD = 0.3  # TODO: add param

# Resources
MEM = 10
TIME = 15

# Plotting
MAP_SIZE = 50
DPI = 100
LEGEND_FONT_SIZE = 9
POINT_SIZE = 3
BACKGROUND_COLOR = 'lightgrey'
INTEREST_COLOR = 'red'
