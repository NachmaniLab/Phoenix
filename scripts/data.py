import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import scanpy as sc
from scripts.consts import ALL_CELLS, CELL_TYPE_COL, NUM_GENES, SEED, TARGET_COL
from scripts.utils import transform_log, re_transform_log
from scripts.output import save_csv
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc.settings.verbosity = 0


def preprocess_expression(expression: pd.DataFrame, preprocessed: bool, num_genes: int = NUM_GENES, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print('Running single-cell preprocessing...')
    adata = sc.AnnData(expression)

    if not preprocessed:
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=5)
        sc.pp.filter_genes(adata, min_counts=500)
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Filter genes using top mean count
    if len(adata.var) > num_genes:
        adata.var['mean_counts'] = adata.X.mean(axis=0)
        adata = adata[:, adata.var_names[np.argsort(adata.var['mean_counts'])[::-1]][:num_genes]]

    return pd.DataFrame(data=adata.X, index=adata.obs_names, columns=adata.var_names)


def reduce_dimension(expression: pd.DataFrame, reduction_method: str, seed: int, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print('Reducing single-cell dimensionality...')

    adata = sc.AnnData(expression)

    sc.tl.pca(adata, random_state=seed)
    if reduction_method == 'umap':
        sc.pp.neighbors(adata, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
    elif reduction_method == 'tsne':
        sc.tl.tsne(adata, random_state=seed)

    return pd.DataFrame(
        adata.obsm[f'X_{reduction_method}'][:, :2],
        columns=[f'{reduction_method}1', f'{reduction_method}2'],
        index=adata.obs_names
    )


def preprocess_data(
        expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        pseudotime: pd.DataFrame,
        reduction: pd.DataFrame | str,
        preprocessed: bool = False,
        exclude_cell_types: list[str] = [],
        exclude_lineages: list[str] = [],
        seed: int = SEED,
        output: str | None = None,
        verbose: bool = True,
    ):
    """
    expression: single-cell raw expression data
    cell_types: single-cell cell type annotations
    pseudotime: single-cell pseudotime values
    reduction: reduction coordinates or method name to use for dimensionality reduction
    preprocessed: whether expression data are already filtered and log-normalized. In this case neither cell filtering nor normalization is applied, only gene filtering if necessary
    exclude_cell_types: list of cell types to exclude from the analysis
    exclude_lineages: list of lineages to exclude from the analysis
    """

    # Filter and normalize
    expression = preprocess_expression(expression, preprocessed, verbose=verbose)

    # Exclude targets
    if cell_types is not None:
        cell_types = cell_types.loc[expression.index]
        exclude_cell_types = [cell_type for cell_type in exclude_cell_types
                              if cell_type in cell_types[CELL_TYPE_COL].tolist()] if exclude_cell_types else []
        if exclude_cell_types:
            if verbose:
                print(f'Excluding cell types: {", ".join(exclude_cell_types)}...')
            cell_types = cell_types[~cell_types[CELL_TYPE_COL].isin(exclude_cell_types)]
            expression = expression.loc[cell_types.index]

    if pseudotime is not None:
        pseudotime = pseudotime.loc[expression.index]
        exclude_lineages = [lineage for lineage in exclude_lineages
                            if lineage in pseudotime.columns] if exclude_lineages else []
        if exclude_lineages:
            if verbose:
                print(f'Excluding lineages: {", ".join(exclude_lineages)}...')
            pseudotime = pseudotime.drop(columns=exclude_lineages)

    # Reduce dimensions
    if isinstance(reduction, str):
        reduction = reduce_dimension(expression, reduction, seed)
    reduction = reduction.loc[expression.index]

    # Save preprocessed data
    if output:
        save_csv(expression, 'expression', output)
        save_csv(cell_types, 'cell_types', output)
        save_csv(pseudotime, 'pseudotime', output)
        save_csv(reduction, 'reduction', output)

    return expression, cell_types, pseudotime, reduction


def scale_expression(expression: pd.DataFrame) -> pd.DataFrame:
    """Scale expression data using standard scaler"""
    return pd.DataFrame(
        StandardScaler().fit_transform(expression),
        index=expression.index, 
        columns=expression.columns
    )


def scale_pseudotime(pseudotime: pd.DataFrame) -> pd.DataFrame:
    "Scale each lineage independently using min-max scaler without missing cells"
    return pd.DataFrame(
        MinMaxScaler().fit_transform(pseudotime),
        index=pseudotime.index, 
        columns=pseudotime.columns
    )


def get_cell_types(cell_types: pd.DataFrame | None) -> list[str]:
    cell_type_list = (cell_types[CELL_TYPE_COL].unique().tolist() + [ALL_CELLS]) if cell_types is not None else []
    return cell_type_list


def get_lineages(pseudotime: pd.DataFrame | None) -> list[str]:
    lineage_list = pseudotime.columns.tolist() if pseudotime is not None else []
    return lineage_list


def sum_gene_expression(gene_set_expression: pd.DataFrame, geometric: bool = False) -> pd.Series:
    if geometric:
        return gene_set_expression.sum() if gene_set_expression.ndim == 1 else gene_set_expression.sum(axis=1)
    untransformed = re_transform_log(gene_set_expression)
    summed = untransformed.sum() if untransformed.ndim == 1 else untransformed.sum(axis=1)
    return transform_log(summed)


def mean_gene_expression(gene_set_expression: pd.DataFrame) -> pd.Series:
    return gene_set_expression.mean(axis=1, skipna=True) if gene_set_expression.ndim > 1 else gene_set_expression.mean(skipna=True)


def calculate_cell_type_effect_size(row, masked_expression, cell_types) -> float:
    target = row[TARGET_COL]
    if target == ALL_CELLS:
        return np.nan
    genes = row['top_genes'].split('; ')

    curr_cells = cell_types[cell_types[CELL_TYPE_COL] == target].index
    other_cells = cell_types[cell_types[CELL_TYPE_COL] != target].index

    curr_sum = mean_gene_expression(masked_expression.loc[curr_cells, genes]).mean()
    other_sum = mean_gene_expression(masked_expression.loc[other_cells, genes]).mean()

    if np.isnan(curr_sum) or np.isnan(other_sum):
        return 0.0
    return curr_sum - other_sum


def calculate_pseudotime_effect_size(row, masked_expression, pseudotime, percentile: float = 0.2, bins: int = 10) -> float:
    target = row[TARGET_COL]
    genes = row['top_genes'].split('; ')

    pseudotime_cells = pseudotime[target].dropna().sort_values(ascending=True).index
    if len(pseudotime_cells) == 0:
        return 0.0
    size = int(np.ceil(len(pseudotime_cells) * percentile))
    
    orig_cells = pseudotime_cells[:size]
    orig_sum = mean_gene_expression(masked_expression.loc[orig_cells, genes]).mean()

    max_change, max_sum = np.nan, np.nan
    last_start = len(pseudotime_cells) - size
    if last_start <= 0:
        return 0.0
    step = max(1, len(pseudotime_cells) // bins)
    for i in range(last_start, 0, -step):
        curr_cells = pseudotime_cells[i:i+size]
        curr_sum = mean_gene_expression(masked_expression.loc[curr_cells, genes]).mean()
        if np.isnan(curr_sum):
            continue
        
        curr_change = abs(curr_sum - orig_sum)
        if np.isnan(max_change) or curr_change > max_change:
            max_change = curr_change
            max_sum = curr_sum

    if np.isnan(orig_sum) or np.isnan(max_sum):
        return 0.0
    return max_sum - orig_sum
