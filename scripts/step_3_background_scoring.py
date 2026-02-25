import os
import time
import pandas as pd
from sklearn.metrics import make_scorer
from scripts.consts import CLASSIFICATION_PREDICTOR, CLASSIFICATION_PREDICTOR_ARGS, METRICS, REGRESSION_PREDICTOR, REGRESSION_PREDICTOR_ARGS, TARGET_COL, BackgroundMode
from scripts.data import get_cell_types, get_lineages, scale_expression, scale_pseudotime
from scripts.prediction import create_cv, get_prediction_score
from scripts.utils import define_background, define_batch_size, remove_outliers, save_step_runtime
from scripts.output import aggregate_batch_results, load_sizes, get_preprocessed_data, save_background_scores


def calculate_background_scores_in_real_mode(
        tmp: str,
        cache: str,
        classification: pd.DataFrame | None = None,
        regression: pd.DataFrame | None = None,
        trim_background: bool = True,
        k_trimming: float = 0.5,
    ) -> None:
    classification = classification if classification is not None else aggregate_batch_results(tmp, 'cell_type_classification')
    regression = regression if regression is not None else aggregate_batch_results(tmp, 'pseudotime_regression')

    if classification is not None:
        for (size, cell_type), subset in classification.groupby(['set_size', TARGET_COL], sort=False):
            background_scores = subset['pathway_score'].to_numpy().tolist()
            background_scores = remove_outliers(background_scores, k=k_trimming) if trim_background else background_scores
            background = define_background(size, background_mode=BackgroundMode.REAL, cell_type=cell_type)
            save_background_scores(background_scores, background, cache)
    
    if regression is not None:
        for (size, lineage), subset in regression.groupby(['set_size', TARGET_COL], sort=False):
            background_scores = subset['pathway_score'].to_numpy().tolist()
            background_scores = remove_outliers(background_scores, k=k_trimming) if trim_background else background_scores
            background = define_background(size, background_mode=BackgroundMode.REAL, lineage=lineage)
            save_background_scores(background_scores, background, cache)


def _get_target_size_pair_batch(sizes: list[int], targets: list[str], batch: int, batch_size: int) -> list[tuple[int, str]]:
    """
    batch: number between 1 and `processes`, or 0 for a single batch
    """
    all_pairs = [(size, target) for size in sizes for target in targets]
    if not batch:
        return all_pairs
    batch_start = (batch - 1) * batch_size
    batch_end = min(batch_start + batch_size, len(all_pairs))
    return all_pairs[batch_start:batch_end]


def calculate_background_scores_in_random_mode(
        sizes: list[int],
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        repeats: int,
        processes: int,
        output: str,
        cache: str,
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str | None = 'cell_types',
        pseudotime: pd.DataFrame | str | None = 'pseudotime',
        trim_background: bool = True,
    ) -> None:
    batch = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))  # index between 1 and `processes`, or 0 for a single batch

    expression = get_preprocessed_data(expression, output)
    cell_types = get_preprocessed_data(cell_types, output)
    pseudotime = get_preprocessed_data(pseudotime, output)

    scaled_expression = scale_expression(expression)
    scaled_pseudotime = scale_pseudotime(pseudotime) if pseudotime is not None else None

    all_cell_types = get_cell_types(cell_types)
    all_lineages = get_lineages(scaled_pseudotime)

    classification_score_function = make_scorer(METRICS[classification_metric], greater_is_better=True)
    regression_score_function = make_scorer(METRICS[regression_metric], greater_is_better=True)

    classification_cv = create_cv(is_regression=False, n_splits=cross_validation)
    regression_cv = create_cv(is_regression=True, n_splits=cross_validation)

    classification_batch_size = define_batch_size(len(sizes) * len(all_cell_types), processes)
    regression_batch_size = define_batch_size(len(sizes) * len(all_lineages), processes)

    cell_type_size_pairs = _get_target_size_pair_batch(sizes, all_cell_types, batch, classification_batch_size)
    lineage_size_pairs = _get_target_size_pair_batch(sizes, all_lineages, batch, regression_batch_size)

    for size, target in cell_type_size_pairs:
        background_scores = []
        for i in range(repeats):
            background_scores.append(get_prediction_score(
                seed=i,
                scaled_expression=scaled_expression,
                cv=classification_cv,
                set_size=size,
                predictor=CLASSIFICATION_PREDICTOR,
                predictor_args=CLASSIFICATION_PREDICTOR_ARGS,  # type: ignore[arg-type]
                score_function=classification_score_function,
                cell_types=cell_types,
                cell_type=target,
            )[0])
        background_scores = remove_outliers(background_scores) if trim_background else background_scores
        background = define_background(
            set_size=size,
            background_mode=BackgroundMode.RANDOM,
            cell_type=target,
            repeats=repeats,
        )
        save_background_scores(background_scores, background, cache)
        
    for size, target in lineage_size_pairs:
        background_scores = []
        for i in range(repeats):
            background_scores.append(get_prediction_score(
                seed=i,
                scaled_expression=scaled_expression,
                cv=regression_cv,
                set_size=size,
                predictor=REGRESSION_PREDICTOR,
                predictor_args=REGRESSION_PREDICTOR_ARGS,  # type: ignore[arg-type]
                score_function=regression_score_function,
                scaled_pseudotime=scaled_pseudotime,
                lineage=target,
            )[0])
        background_scores = remove_outliers(background_scores) if trim_background else background_scores
        background = define_background(
            set_size=size,
            background_mode=BackgroundMode.RANDOM,
            lineage=target,
            repeats=repeats,
        )
        save_background_scores(background_scores, background, cache)


def calculate_background_scores(
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        repeats: int,
        processes: int,
        output: str,
        tmp: str,
        cache: str,
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        classification: pd.DataFrame | None = None,
        regression: pd.DataFrame | None = None,
        sizes: list[int] | None = None,
        background_mode: BackgroundMode | None = None,
        trim_background: bool = True,
        verbose: bool = True,
    ) -> None:
    if sizes is None or background_mode is None:
        sizes, background_mode = load_sizes(output)

    step_start = time.time()
    batch = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    match background_mode:
        case BackgroundMode.REAL:
            if verbose:
                print(f'Calculating background scores for real mode...')
            calculate_background_scores_in_real_mode(
                tmp=tmp,
                cache=cache,
                classification=classification,
                regression=regression,
                trim_background=trim_background,
            )
        case BackgroundMode.RANDOM:
            if verbose:
                print(f'Calculating background scores for random mode...')
            calculate_background_scores_in_random_mode(
                sizes=sizes,
                classification_metric=classification_metric,
                regression_metric=regression_metric,
                cross_validation=cross_validation,
                repeats=repeats,
                processes=processes,
                output=output,
                cache=cache,
                expression=expression,
                cell_types=cell_types,
                pseudotime=pseudotime,
                trim_background=trim_background,
            )

    save_step_runtime(tmp, 'step3', time.time() - step_start, batch)

