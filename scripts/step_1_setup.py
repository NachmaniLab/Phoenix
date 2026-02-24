import time
import pandas as pd
from scripts.consts import BackgroundMode
from scripts.data import preprocess_data
from scripts.backgrounds import define_sizes, set_background_mode
from scripts.pathways import get_gene_sets
from scripts.utils import define_batch_size, save_step_runtime, str2enum
from scripts.output import read_raw_data


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
        random_sizes: list[int],
        repeats: int,
        set_fraction: float,
        min_set_size: int,
        seed: int,
        processes: int,
        output: str,
        tmp: str,
        return_data: bool = False,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[int]] | None:

    step_start = time.time()

    expression, cell_types, pseudotime, reduction = read_raw_data(expression, cell_types, pseudotime, reduction)
    expression, cell_types, pseudotime, reduction = preprocess_data(expression, cell_types, pseudotime, reduction, preprocessed=preprocessed, exclude_cell_types=exclude_cell_types, exclude_lineages=exclude_lineages, seed=seed, output=output, verbose=verbose)
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism, expression.columns, min_set_size, output)  # type: ignore[attr-defined]
    
    background_mode = str2enum(BackgroundMode, background_mode)
    background_mode = set_background_mode(background_mode, len(gene_sets))  # type: ignore[arg-type]
    sizes = define_sizes(background_mode, gene_sets, set_fraction, min_set_size, repeats, random_sizes, output)

    if verbose:
        print(f'Background mode: {background_mode.name}')
        print(f'Running experiments for {len(gene_sets)} gene annotations with batch size of {define_batch_size(len(gene_sets), processes)}...')

    save_step_runtime(tmp, 'step1', time.time() - step_start)

    if return_data:
        return expression, cell_types, pseudotime, reduction, gene_sets, sizes
    return None
