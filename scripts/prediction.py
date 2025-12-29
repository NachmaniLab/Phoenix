import os
import warnings
import sys
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as stats
from scripts.data import get_cell_types, get_lineages, calculate_cell_type_effect_size, calculate_pseudotime_effect_size
from scripts.train import get_train_data, train
from scripts.consts import CLASSIFIERS, REGRESSORS, CLASSIFIER_ARGS, REGRESSOR_ARGS, SIZES, BackgroundMode
from scripts.utils import define_background, define_set_size, remove_outliers
from scripts.output import load_background_scores, save_background_scores, summarise_result, save_csv, get_preprocessed_data


def get_prediction_score(
        scaled_expression: pd.DataFrame,
        predictor: str,
        metric: str,
        seed: int,
        gene_set: list[str] | None = None,
        set_size: int | None = None,
        feature_selection: str | None = None,
        cross_validation: int | None = None,
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
    ) -> tuple[float, list[str], list[float] | None]:

    X, y, selected_genes, gene_importances = get_train_data(
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

    is_regression = scaled_pseudotime is not None
    predictor = REGRESSORS[predictor] if is_regression else CLASSIFIERS[predictor]
    predictor_args = REGRESSOR_ARGS[predictor] if is_regression else CLASSIFIER_ARGS[predictor]
    
    score = train(
        X=X,
        y=y,
        predictor=predictor,
        predictor_args=predictor_args,
        metric=metric,
        cross_validation=cross_validation,
        seed=seed
    )

    return score, selected_genes, gene_importances


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


def run_comparison(
        scaled_expression: pd.DataFrame,
        gene_set: list[str],
        predictor: str,
        metric: str,
        set_size: int,
        feature_selection: str | None,
        cross_validation: int,
        repeats: int,
        seed: int,
        distribution: str,
        cell_types: pd.DataFrame | None = None,
        scaled_pseudotime: pd.DataFrame | None = None,
        cell_type: str | None = None,
        lineage: str | None = None,
        trim_background: bool = True,
        cache: str | None = None
    ) -> tuple[float, float, list[float], list[str], list[float] | None]:

    prediction_args = {
        'scaled_expression': scaled_expression,
        'predictor': predictor,
        'metric': metric,
        'cross_validation': cross_validation,
        'set_size': set_size,
        'cell_types': cell_types,
        'scaled_pseudotime': scaled_pseudotime,
        'cell_type': cell_type,
        'lineage': lineage,
    }

    # Pathway of interest
    pathway_score, top_genes, gene_importances = get_prediction_score(seed=seed, gene_set=gene_set, feature_selection=feature_selection, **prediction_args)

    # Background
    background = define_background(set_size, BackgroundMode.RANDOM, cell_type, lineage, repeats)
    background_scores = load_background_scores(background, cache)
    if not background_scores:
        for i in range(repeats):
            background_scores.append(get_prediction_score(seed=i, **prediction_args)[0])
        if trim_background:
            background_scores = remove_outliers(background_scores)
        save_background_scores(background_scores, background, cache)

    # Compare scores
    p_value = compare_scores(pathway_score, background_scores, distribution)

    return p_value, pathway_score, background_scores, top_genes, gene_importances


def get_gene_set_batch(gene_sets: dict[str, list[str]], batch: int, batch_size: int) -> dict[str, list[str]]:
    """
    batch: number between 1 and `processes`, or 0 for a single batch
    """
    if not batch or batch is None:
        return gene_sets
    batch_start = (batch - 1) * batch_size
    batch_end = min(batch_start + batch_size, len(gene_sets))
    set_names = list(gene_sets.keys())[batch_start:batch_end]
    return {set_name: gene_sets[set_name] for set_name in set_names}


def run_batch(
        batch: int | None,
        batch_gene_sets: dict[str, list[str]],
        scaled_expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        scaled_pseudotime: pd.DataFrame,
        feature_selection: str,
        set_fraction: float,
        min_set_size: int,
        classifier: str,
        regressor: str,
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        repeats: int,
        seed: int,
        distribution: str,
        output: str,
        cache: str,
        effect_size_threshold: float = 0.3,
        verbose: bool = True,
    ) -> None:
    """
    output: main output path for a single batch and temp output path for many batches
    batch: number between 1 and `processes`, or None for a single batch
    """
    if batch_gene_sets is None:
        return

    classification_results, regression_results = [], []
    all_cell_types, all_lineages = get_cell_types(cell_types), get_lineages(scaled_pseudotime)

    logger = f'Batch {batch}: ' if batch else ''    
    for i, (set_name, gene_set) in tqdm(
        enumerate(batch_gene_sets.items()),
        total=len(batch_gene_sets),
        desc='Batch',
        ncols=80,
        ascii=True,
        file=sys.stdout if batch else None,
        disable=not verbose,
    ):
        if verbose:

            print(f'\n{logger}Pathway {i + 1}/{len(batch_gene_sets)}: {set_name}', flush=True)
            sys.stdout.flush()

        set_size = define_set_size(len(gene_set), set_fraction, min_set_size, all_sizes=SIZES)
        task_args = {
            'scaled_expression': scaled_expression, 'gene_set': gene_set,
            'set_size': set_size, 'feature_selection': feature_selection,
            'cross_validation': cross_validation, 'repeats': repeats,
            'seed': seed, 'distribution': distribution, 'cache': cache
        }

        # Cell-type classification
        random.shuffle(all_cell_types)
        for cell_type in all_cell_types:
            p_value, pathway_score, background_scores, top_genes, gene_importances = run_comparison(
                predictor=classifier, metric=classification_metric,
                cell_types=cell_types, cell_type=cell_type,
                **task_args
            )
            classification_results.append(summarise_result(
                cell_type, set_name, top_genes, gene_importances, set_size, feature_selection, classifier, classification_metric,
                cross_validation, repeats, distribution, seed, pathway_score, background_scores, p_value
            ))
        
        # Pseudo-time regression
        random.shuffle(all_lineages)
        for lineage in all_lineages:
            p_value, pathway_score, background_scores, top_genes, gene_importances = run_comparison(
                predictor=regressor, metric=regression_metric,
                scaled_pseudotime=scaled_pseudotime, lineage=lineage,
                **task_args
            )

            regression_results.append(summarise_result(
                lineage, set_name, top_genes, gene_importances, set_size, feature_selection, regressor, regression_metric,
                cross_validation, repeats, distribution, seed, pathway_score, background_scores, p_value
            ))

    classification = pd.DataFrame(classification_results)
    regression = pd.DataFrame(regression_results)

    # Add effect size
    main_output_path = output if not batch else os.path.dirname(output)
    expression = get_preprocessed_data('expression', main_output_path)  # not scaled
    masked_expression = expression.mask(expression <= effect_size_threshold)
    classification['effect_size'] = classification.apply(calculate_cell_type_effect_size, axis=1, masked_expression=masked_expression, cell_types=cell_types)
    regression['effect_size'] = regression.apply(calculate_pseudotime_effect_size, axis=1, masked_expression=masked_expression, pseudotime=scaled_pseudotime)

    # Save results
    info = f'_batch{batch}' if batch else ''
    save_csv(classification, f'cell_type_classification{info}', output, keep_index=False)
    save_csv(regression, f'pseudotime_regression{info}', output, keep_index=False)
