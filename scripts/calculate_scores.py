import os
import sys
import pandas as pd
from tqdm import tqdm
from scripts.consts import TARGET_COL, BackgroundMode
from scripts.data import calculate_cell_type_effect_size, calculate_pseudotime_effect_size, get_cell_types, get_lineages, scale_expression, scale_pseudotime
from scripts.prediction import get_prediction_score, get_gene_set_batch
from scripts.utils import convert_to_str, define_background, define_batch_size, define_set_size, remove_outliers
from scripts.output import aggregate_batch_results, load_sizes, get_preprocessed_data, read_gene_sets, save_background_scores, save_csv


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
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        gene_sets: dict[str, list[str]] | str = 'gene_sets',
        sizes: list[int] | None = None,
        effect_size_threshold: float = 0.3,
        verbose: bool = True,
    ) -> None | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pathway scores for a single batch of pathways.
    """
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

        set_size = define_set_size(len(gene_set), set_fraction, min_set_size, all_sizes=sizes)
        
        # Cell-type classification
        for cell_type in all_cell_types:
            pathway_score, top_genes, gene_importances = get_prediction_score(
                scaled_expression=scaled_expression,
                predictor=classifier,
                metric=classification_metric,
                seed=seed,
                gene_set=gene_set,
                cross_validation=cross_validation,
                set_size=set_size,
                feature_selection=feature_selection,
                cell_types=cell_types,
                cell_type=cell_type,
            )
            result = {
                TARGET_COL: cell_type,
                'set_name': set_name,
                'top_genes': top_genes,
                'gene_importances': gene_importances,
                'set_size': set_size,
                'pathway_score': pathway_score,
            }
            classification_results.append({key: convert_to_str(value) for key, value in result.items()})
        
        # Pseudo-time regression
        for lineage in all_lineages:
            pathway_score, top_genes, gene_importances = get_prediction_score(
                scaled_expression=scaled_expression,
                predictor=regressor,
                metric=regression_metric,
                seed=seed,
                gene_set=gene_set,
                cross_validation=cross_validation,
                set_size=set_size,
                feature_selection=feature_selection,
                scaled_pseudotime=scaled_pseudotime,
                lineage=lineage,
            )
            result = {
                TARGET_COL: lineage,
                'set_name': set_name,
                'top_genes': top_genes,
                'gene_importances': gene_importances,
                'set_size': set_size,
                'pathway_score': pathway_score,
            }
            regression_results.append({key: convert_to_str(value) for key, value in result.items()})
    
    classification = pd.DataFrame(classification_results)
    regression = pd.DataFrame(regression_results)

    # Add effect size
    masked_expression = expression.mask(expression <= effect_size_threshold)  # not scaled
    if not classification.empty:
        classification['effect_size'] = classification.apply(calculate_cell_type_effect_size, axis=1, masked_expression=masked_expression, cell_types=cell_types)
    if not regression.empty:
        regression['effect_size'] = regression.apply(calculate_pseudotime_effect_size, axis=1, masked_expression=masked_expression, pseudotime=scaled_pseudotime)

    # Save or return results
    if batch:
        save_csv(classification, f'cell_type_classification_batch{batch}', tmp, keep_index=False)
        save_csv(regression, f'pseudotime_regression_batch{batch}', tmp, keep_index=False)
        return None
    else:
        return classification, regression


def calculate_background_scores_in_real_mode(
        tmp: str,
        cache: str,
        classification: pd.DataFrame | None = None,
        regression: pd.DataFrame | None = None,
        trim_background: bool = True,
    ) -> None:
    classification = classification if classification is not None else aggregate_batch_results(tmp, 'cell_type_classification')
    regression = regression if regression is not None else aggregate_batch_results(tmp, 'pseudotime_regression')

    if classification is not None:
        for (size, cell_type), subset in classification.groupby(['set_size', TARGET_COL], sort=False):
            background_scores = subset['pathway_score'].to_numpy().tolist()
            background_scores = remove_outliers(background_scores) if trim_background else background_scores
            background = define_background(size, background_mode=BackgroundMode.REAL, cell_type=cell_type)
            save_background_scores(background_scores, background, cache)
    
    if regression is not None:
        for (size, lineage), subset in regression.groupby(['set_size', TARGET_COL], sort=False):
            background_scores = subset['pathway_score'].to_numpy().tolist()
            background_scores = remove_outliers(background_scores) if trim_background else background_scores
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
        classifier: str,
        regressor: str,
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

    batch_size = define_batch_size(len(sizes) * (len(all_cell_types) + len(all_lineages)), processes)
    target_size_pairs = _get_target_size_pair_batch(sizes, all_cell_types + all_lineages, batch, batch_size)

    for size, target in target_size_pairs:
        is_classification = target in all_cell_types
        background_scores = []
        for i in range(repeats):
            background_scores.append(get_prediction_score(
                seed=i,
                scaled_expression=scaled_expression,
                cross_validation=cross_validation,
                set_size=size,
                predictor=classifier if is_classification else regressor,
                metric=classification_metric if is_classification else regression_metric,
                cell_types=cell_types if is_classification else None,
                scaled_pseudotime=scaled_pseudotime if not is_classification else None,
                cell_type=target if is_classification else None,
                lineage=target if not is_classification else None,
            )[0])
        background_scores = remove_outliers(background_scores) if trim_background else background_scores
        background = define_background(
            set_size=size,
            background_mode=BackgroundMode.RANDOM,
            cell_type=target if is_classification else None,
            lineage=target if not is_classification else None,
            repeats=repeats,
        )
        save_background_scores(background_scores, background, cache)
        

def calculate_background_scores(
        classifier: str,
        regressor: str,
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
    ) -> None:
    if sizes is None or background_mode is None:
        sizes, background_mode = load_sizes(output)

    match background_mode:
        case BackgroundMode.REAL:
            calculate_background_scores_in_real_mode(
                tmp=tmp,
                cache=cache,
                classification=classification,
                regression=regression,
                trim_background=trim_background,
            )
        case BackgroundMode.RANDOM:
            calculate_background_scores_in_random_mode(
                sizes=sizes,
                classifier=classifier,
                regressor=regressor,
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

