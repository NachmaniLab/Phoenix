import time as runtime
from scripts.args import get_run_args
from scripts.consts import BackgroundMode
from scripts.pipeline import run_setup_cmd, run_pathway_scoring_cmd, run_background_scoring_cmd, run_aggregation_cmd
from scripts.step_1_setup import setup
from scripts.step_2_pathway_scoring import calculate_pathway_scores
from scripts.step_3_background_scoring import calculate_background_scores
from scripts.step_4_aggregation import aggregate


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
        effect_size_threshold: float | None,
        corrected_effect_size: bool,
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

        # Pathway scoring
        pathway_scoring_args = {
            'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
            'classifier': classifier, 'regressor': regressor,
            'classification_metric': classification_metric, 'regression_metric': regression_metric,
            'cross_validation': cross_validation, 'seed': seed, 'processes': processes,
            'output': output, 'tmp': tmp, 'effect_size_threshold': effect_size_threshold,
        }
        pathway_scoring_job_id = run_pathway_scoring_cmd(pathway_scoring_args, processes, mem, time, tmp, setup_job_id)

        # Background scoring
        background_scoring_args = {
            'classifier': classifier, 'regressor': regressor,
            'classification_metric': classification_metric, 'regression_metric': regression_metric,
            'cross_validation': cross_validation, 'repeats': repeats, 'processes': processes,
            'output': output, 'tmp': tmp, 'cache': cache,
        }
        background_scoring_job_id = run_background_scoring_cmd(background_scoring_args, processes, mem, time, tmp, pathway_scoring_job_id)

        # Aggregation
        aggregation_args = {
            'output': output, 'tmp': tmp, 'cache': cache,
            'background_mode': background_mode, 'distribution': distribution, 'repeats': repeats,
            'corrected_effect_size': corrected_effect_size, 'start_time': start_time
        }
        run_aggregation_cmd(aggregation_args, processes, tmp, background_scoring_job_id)
    
    else:
        expression, cell_types, pseudotime, reduction, gene_sets, sizes = setup(
            expression, cell_types, pseudotime, reduction,
            preprocessed, exclude_cell_types, exclude_lineages,
            pathway_database, custom_pathways, organism,
            background_mode, repeats, set_fraction, min_set_size,
            seed, processes, output, return_data=True, verbose=verbose
        )  # type: ignore[misc]
        classification, regression = calculate_pathway_scores(
            feature_selection, set_fraction, min_set_size,
            classifier, regressor, classification_metric, regression_metric,
            cross_validation, seed, processes,
            output, tmp, effect_size_threshold,
            expression, cell_types, pseudotime,
            gene_sets, sizes, verbose=verbose
        )  # type: ignore[misc]
        calculate_background_scores(
            classifier, regressor, classification_metric, regression_metric,
            cross_validation, repeats, processes,
            output, tmp, cache,
            expression, cell_types, pseudotime,
            classification, regression,
            sizes, background_mode, verbose=verbose
        )
        aggregate(
            output, tmp, cache,
            background_mode, distribution, repeats, corrected_effect_size,
            classification, regression, start_time, verbose=verbose
        )
        

if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
