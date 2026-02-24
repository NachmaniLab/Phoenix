import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import make_scorer
from scripts.consts import CLASSIFIER_ARGS, CLASSIFIERS, EFFECT_SIZE_THRESHOLD, METRICS, REGRESSOR_ARGS, REGRESSORS, TARGET_COL
from scripts.data import calculate_cell_type_effect_size, calculate_pseudotime_effect_size, get_cell_types, get_lineages, scale_expression, scale_pseudotime
from scripts.prediction import create_cv, get_prediction_score
from scripts.utils import convert_to_str, define_batch_size, define_set_size, save_step_runtime
from scripts.output import load_sizes, get_preprocessed_data, read_gene_sets, save_csv


def get_gene_set_batch(gene_sets: dict[str, list[str]], batch: int, batch_size: int) -> dict[str, list[str]]:
    """
    batch: number between 1 and `processes`, or 0 for a single batch
    """
    if not batch:
        return gene_sets
    batch_start = (batch - 1) * batch_size
    batch_end = min(batch_start + batch_size, len(gene_sets))
    set_names = list(gene_sets.keys())[batch_start:batch_end]
    return {set_name: gene_sets[set_name] for set_name in set_names}


def calculate_pathway_scores(
        feature_selection: str,
        set_fraction: float,
        min_set_size: int,
        classifier: str,
        regressor: str,
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        seed: int,
        processes: int,
        output: str,
        tmp: str,
        effect_size_threshold: float | None,
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        gene_sets: dict[str, list[str]] | str = 'gene_sets',
        sizes: list[int] | None = None,
        verbose: bool = True,
    ) -> None | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pathway scores for a single batch of pathways.
    """
    step_start = time.time()
    batch = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))  # index between 1 and `processes`, or 0 for a single batch

    expression = get_preprocessed_data(expression, output)
    cell_types = get_preprocessed_data(cell_types, output)
    pseudotime = get_preprocessed_data(pseudotime, output)

    scaled_expression = scale_expression(expression)
    scaled_pseudotime = scale_pseudotime(pseudotime) if pseudotime is not None else None

    gene_sets = read_gene_sets(output, gene_sets)
    batch_size = define_batch_size(len(gene_sets), processes)
    batch_gene_sets = get_gene_set_batch(gene_sets, batch, batch_size)

    sizes = load_sizes(output)[0] if sizes is None else sizes

    classification_results, regression_results = [], []
    all_cell_types, all_lineages = get_cell_types(cell_types), get_lineages(scaled_pseudotime)

    classification_score_function = make_scorer(METRICS[classification_metric], greater_is_better=True)
    regression_score_function = make_scorer(METRICS[regression_metric], greater_is_better=True)

    classification_predictor = CLASSIFIERS[classifier]
    regression_predictor = REGRESSORS[regressor]

    classifier_args = CLASSIFIER_ARGS[classification_predictor]
    regressor_args = REGRESSOR_ARGS[regression_predictor]

    classification_cv = create_cv(is_regression=False, n_splits=cross_validation)
    regression_cv = create_cv(is_regression=True, n_splits=cross_validation)

    logger = f'Batch {batch}: ' if batch else ''    
    for i, (set_name, gene_set) in tqdm(
        enumerate(batch_gene_sets.items()),
        total=len(batch_gene_sets),
        desc='Batch',
        ncols=80,
        ascii=True,
        smoothing=0.9,
        file=sys.stdout if batch else None,
        disable=not verbose,
    ):
        if verbose and i % 10 == 0:
            print(f'\n{logger}Pathway {i + 1}/{len(batch_gene_sets)}: {set_name}', flush=True)
            sys.stdout.flush()

        set_size = define_set_size(len(gene_set), set_fraction, min_set_size, all_sizes=sizes)
        
        # Cell-type classification
        for cell_type in all_cell_types:
            pathway_score, top_genes, gene_importances = get_prediction_score(
                scaled_expression=scaled_expression,
                predictor=classification_predictor,
                predictor_args=classifier_args,  # type: ignore[arg-type]
                score_function=classification_score_function,
                seed=seed,
                gene_set=gene_set,
                cv=classification_cv,
                set_size=set_size,
                feature_selection=feature_selection,
                cell_types=cell_types,
                cell_type=cell_type,
            )
            result = {
                TARGET_COL: cell_type,
                'set_name': set_name,
                'top_genes': convert_to_str(top_genes),
                'gene_importances': convert_to_str(gene_importances),
                'set_size': set_size,
                'pathway_score': pathway_score,
            }
            classification_results.append(result)
        
        # Pseudo-time regression
        for lineage in all_lineages:
            pathway_score, top_genes, gene_importances = get_prediction_score(
                scaled_expression=scaled_expression,
                predictor=regression_predictor,
                predictor_args=regressor_args,  # type: ignore[arg-type]
                score_function=regression_score_function,
                seed=seed,
                gene_set=gene_set,
                cv=regression_cv,
                set_size=set_size,
                feature_selection=feature_selection,
                scaled_pseudotime=scaled_pseudotime,
                lineage=lineage,
            )
            result = {
                TARGET_COL: lineage,
                'set_name': set_name,
                'top_genes': convert_to_str(top_genes),
                'gene_importances': convert_to_str(gene_importances),
                'set_size': set_size,
                'pathway_score': pathway_score,
            }
            regression_results.append(result)
    
    classification = pd.DataFrame(classification_results)
    regression = pd.DataFrame(regression_results)

    # Add effect size
    if effect_size_threshold is not None:
        masked_expression = expression.mask(expression <= effect_size_threshold)  # not scaled
    else:
        masked_expression = expression
    if not classification.empty:
        classification['effect_size'] = calculate_cell_type_effect_size(classification, masked_expression, cell_types)
    if not regression.empty:
        regression['effect_size'], regression['most_diff_pseudotime'] = calculate_pseudotime_effect_size(regression, masked_expression, scaled_pseudotime)

    save_step_runtime(tmp, 'step2', time.time() - step_start, batch)

    # Save or return results
    if batch:
        save_csv(classification, f'cell_type_classification_batch{batch}', tmp, keep_index=False)
        save_csv(regression, f'pseudotime_regression_batch{batch}', tmp, keep_index=False)
        return None
    else:
        return classification, regression
