import os
import time as runtime
import pandas as pd
from scripts.args import get_run_args
from scripts.consts import BackgroundMode
from scripts.data import preprocess_data, scale_expression, scale_pseudotime
from scripts.backgrounds import define_sizes, set_background_mode
from scripts.pathways import get_gene_sets
from scripts.computation import run_setup_cmd, run_experiments_cmd, run_aggregation_cmd
from scripts.prediction import run_batch, get_gene_set_batch
from scripts.utils import define_batch_size
from scripts.output import read_raw_data, aggregate_result, get_preprocessed_data, read_gene_sets
from scripts.visualization import plot


def setup(
        expression: str,
        cell_types: str,
        pseudotime: str,
        reduction: str,
        preprocessed: bool,
        exclude_cell_types: list[str],
        exclude_lineages: list[str],
        pathway_database: list[str],
        custom_pathways: list[str],
        organism: str,
        background_mode: BackgroundMode | str,
        repeats: int,
        set_fraction: float,
        min_set_size: int,
        seed: int,
        processes: int,
        output: str,
        return_data: bool = False,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[int]] | None:

    expression, cell_types, pseudotime, reduction = read_raw_data(expression, cell_types, pseudotime, reduction)
    expression, cell_types, pseudotime, reduction = preprocess_data(expression, cell_types, pseudotime, reduction, preprocessed=preprocessed, exclude_cell_types=exclude_cell_types, exclude_lineages=exclude_lineages, seed=seed, output=output, verbose=verbose)
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism, expression.columns, min_set_size, output)  # type: ignore[attr-defined]
    
    background_mode = BackgroundMode[background_mode] if isinstance(background_mode, str) else background_mode
    background_mode = set_background_mode(background_mode, repeats, len(gene_sets))
    sizes = define_sizes(background_mode, gene_sets, set_fraction, min_set_size, output)

    if verbose:
        print(f'Background mode: {background_mode.name}')
        print(f'Running experiments for {len(gene_sets)} gene annotations with batch size of {define_batch_size(len(gene_sets), processes)}...')

    if return_data:
        return expression, cell_types, pseudotime, reduction, gene_sets, sizes
    return None


def run_experiments(
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
        processes: int,
        output: str,
        tmp: str,
        cache: str,
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        gene_sets: dict[str, list[str]] | str = 'gene_sets',
        verbose: bool = True,
    ) -> None:
    """
    Run experiments for a single batch of gene sets.
    output: main output path
    """
    batch = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))  # index between 1 and `processes`, or None for a single batch

    expression = get_preprocessed_data(expression, output)
    cell_types = get_preprocessed_data(cell_types, output)
    pseudotime = get_preprocessed_data(pseudotime, output)

    scaled_expression = scale_expression(expression)
    scaled_pseudotime = scale_pseudotime(pseudotime) if pseudotime is not None else None

    gene_sets = read_gene_sets(output, gene_sets)
    batch_size = define_batch_size(len(gene_sets), processes)
    batch_gene_sets = get_gene_set_batch(gene_sets, batch, batch_size)      

    run_batch(
        batch, batch_gene_sets, scaled_expression, cell_types, scaled_pseudotime,
        feature_selection, set_fraction, min_set_size,
        classifier, regressor, classification_metric, regression_metric,
        cross_validation, repeats, seed, distribution,
        tmp if batch else output, cache,
        verbose=verbose,
    )


def summarize(
        output: str,
        tmp: str | None = None,
        start_time: float | None = None,
        verbose: bool = True,
    ) -> None:
    if verbose:
        print('Aggregating results...')
    aggregate_result('cell_type_classification', output, tmp)
    aggregate_result('pseudotime_regression', output, tmp)

    if verbose:
        print('Plotting results...')
    plot(output)

    if verbose and start_time is not None:
        elapsed = runtime.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        breakpoint()
        print(f"Runtime: {hours}h {minutes}m {seconds}s")


def run_tool(
        expression: str,
        cell_types: str,
        pseudotime: str,
        reduction: str,
        preprocessed: bool,
        exclude_cell_types: list[str],
        exclude_lineages: list[str],
        organism: str,
        pathway_database: list[str],
        custom_pathways: list[str],
        feature_selection: str,
        set_fraction: float,
        min_set_size: int,
        classifier: str,
        regressor: str,
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        background_mode: BackgroundMode,
        repeats: int,
        seed: int,
        distribution: str,
        processes: int,
        mem: int,
        time: int,
        output: str,
        cache: str,
        tmp: str,
        verbose: bool = True,
    ) -> None:
    start_time = runtime.time()

    if processes:
        # Setup
        setup_args = {
            'expression': expression, 'cell_types': cell_types, 'pseudotime': pseudotime, 'reduction': reduction,
            'preprocessed': preprocessed, 'exclude_cell_types': exclude_cell_types, 'exclude_lineages': exclude_lineages,
            'pathway_database': pathway_database, 'custom_pathways': custom_pathways, 'organism': organism,
            'background_mode': background_mode, 'repeats': repeats, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
            'seed': seed, 'processes': processes, 'output': output
        }
        setup_job_id = run_setup_cmd(setup_args, tmp)

        # Experiments
        exp_args = {
            'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
            'classifier': classifier, 'regressor': regressor, 'classification_metric': classification_metric, 'regression_metric': regression_metric,
            'cross_validation': cross_validation, 'repeats': repeats, 'seed': seed, 'distribution': distribution,
            'processes': processes, 'output': output, 'tmp': tmp, 'cache': cache,
        }
        exp_job_id = run_experiments_cmd(setup_job_id, mem, time, exp_args, tmp)

        # Aggregation
        run_aggregation_cmd(exp_job_id, processes, output, tmp, start_time)
    
    else:
        expression, cell_types, pseudotime, reduction, gene_sets, sizes = setup(expression, cell_types, pseudotime, reduction, preprocessed, exclude_cell_types, exclude_lineages, pathway_database, custom_pathways, organism, background_mode, repeats, set_fraction, min_set_size, seed, processes, output, return_data=True, verbose=verbose)  # type: ignore[misc]
        run_experiments(feature_selection, set_fraction, min_set_size, classifier, regressor, classification_metric, regression_metric, cross_validation, repeats, seed, distribution, processes, output, tmp, cache, expression, cell_types, pseudotime, gene_sets, verbose=verbose)
        summarize(output, start_time=start_time, verbose=verbose)


if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
