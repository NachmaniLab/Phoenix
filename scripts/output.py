import os, yaml, glob  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scripts.consts import TARGET_COL, CELL_TYPE_COL
from scripts.utils import make_valid_filename, convert_to_str, convert_from_str, adjust_p_value, correct_effect_size


# TODO: add methods to return paths of cache, reports and batch results (if needed)


def read_csv(path: str, index_col: int = 0, dtype=None, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f'Reading file at {path}...')
    try:
        return pd.read_csv(path, index_col=index_col, dtype=dtype)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path '{path}'")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{path}' is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Parsing failed for file '{path}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading '{path}': {str(e)}")


def save_csv(data: list[dict] | pd.DataFrame | dd.DataFrame | None, title: str, output_path: str, keep_index: bool = True) -> None:
    # TODO: if already exists
     
    if data is None: return
    if isinstance(data, list):
        if not data: return

    if isinstance(data, dd.DataFrame):
        data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), single_file=True, index=keep_index)

    else:
        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), index=keep_index)


def read_raw_data(expression: str, cell_types: str | None, pseudotime: str | None, reduction: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression = read_csv(expression)
    cell_types = read_csv(cell_types).loc[expression.index] if cell_types else None
    if cell_types is not None:
        cell_types = cell_types.rename(columns={cell_types.columns[0]: CELL_TYPE_COL})
    pseudotime = read_csv(pseudotime).loc[expression.index] if pseudotime else None 
    if os.path.exists(reduction):
        reduction = read_csv(reduction).loc[expression.index]
    return expression, cell_types, pseudotime, reduction


def load_background_scores(background: str, cache_path: str | None = None, verbose: bool = False):
    background = make_valid_filename(background).lower()
    if cache_path and os.path.exists(f'{cache_path}/{background}.yml') and os.path.getsize(f'{cache_path}/{background}.yml') > 0:
        if verbose:
            print(f'Loading background {background} from cache...')
        with open(f'{cache_path}/{background}.yml', 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    return []


def save_background_scores(background_scores: list[float], background: str, cache_path: str | None = None, verbose: bool = False):
    if cache_path:
        background = make_valid_filename(background).lower()
        if verbose:
            print(f'Saving background {background} in cache...')
        with open(f'{cache_path}/{background}.yml', 'w') as file:
            yaml.dump(background_scores, file)


def read_gene_sets(output_path: str, title: str | dict = 'gene_sets') -> dict[str, list[str]]:
    if isinstance(title, dict):
        return title
    path = output_path if '.csv' in output_path else os.path.join(output_path, f'{make_valid_filename(title)}.csv') 
    df = read_csv(path, index_col=False, dtype=str)
    return {column: df[column].dropna().tolist() for column in df.columns}


def save_gene_sets(gene_sets: dict[str, list[str]], output_path: str, title: str = 'gene_sets', by_set: bool = False) -> None:
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gene_sets.items()]))
    save_csv(df, title, output_path, keep_index=False)

    if by_set:
        for col in df.columns:
            save_csv(pd.DataFrame(df[col]).dropna(), col, output_path, keep_index=False)


def summarise_result(target, set_name, top_genes, set_size, feature_selection, predictor, metric, cross_validation, repeats, distribution, seed, pathway_score, background_scores: list[float], p_value):
    result = {
        TARGET_COL: target,
        'set_name': set_name,
        'top_genes': top_genes,
        'set_size': set_size,
        'feature_selection': feature_selection if feature_selection else 'None',
        'predictor': predictor,
        'metric': metric,
        'cross_validation': cross_validation,
        'repeats': repeats,
        'distribution': distribution,
        'seed': seed,
        'pathway_score': pathway_score,
        'background_scores': background_scores,
        'background_score_mean': np.mean(background_scores),
        'p_value': p_value,
    }
    return {key: convert_to_str(value) for key, value in result.items()}


def read_results(title: str, output_path: str, index_col: int | None = None, raise_err: bool = False) -> pd.DataFrame | None:
    try:
        title = f'{title}.csv' if '.csv' not in title else title
        return read_csv(os.path.join(output_path, f'{make_valid_filename(title)}'), index_col=index_col)
    except Exception as e:
        if raise_err:
            raise e
        return None


def get_preprocessed_data(data: pd.DataFrame | str, output_path: str):
    if isinstance(data, str):
        data = read_results(data, output_path, index_col=0, raise_err=True)
    return data


def aggregate_result(result_type: str, output: str, tmp: str | None) -> pd.DataFrame | None:
    df = read_results(result_type, output)
    if df is not None:  # if run was in a single batch or results already aggregated
        if 'fdr' not in df.columns:
            df['fdr'] = adjust_p_value(df['p_value'])
            df['effect_size'] = correct_effect_size(df['effect_size'], df[TARGET_COL])
            save_csv(df, result_type, output, keep_index=False)
        return df
    
    dfs = []
    for path in glob.glob(os.path.join(tmp, f'{result_type}_batch*.csv')):  # type: ignore[arg-type]
        df = read_results(os.path.basename(path), tmp, index_col=None)  # type: ignore[arg-type]
        if df is not None:
            dfs.append(df)

    if not dfs:  # if result type is missing
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df['fdr'] = adjust_p_value(df['p_value'])
    df['effect_size'] = correct_effect_size(df['effect_size'], df[TARGET_COL])
    save_csv(dd.from_pandas(df, npartitions=1), result_type, output, keep_index=False)
    return df


def get_experiment(results: pd.DataFrame | str, output_path: str, set_name: str | None = None, target: str | None = None) -> pd.DataFrame | dict:
    if isinstance(results, str):
        results = read_results(results, output_path)
    if set_name and results is not None:
        results = results[results['set_name'] == set_name]
    if target and results is not None:
        results = results[results[TARGET_COL] == target]
    
    if set_name and target and results is not None:
        results = results.iloc[0]
        return {key: convert_from_str(results[key]) for key in results.index}

    return results


def save_plot(title: str, output: str | None = None, format: str = 'png') -> None:
    plt.tight_layout()
    if output:
        create_dir(output)
        file_path = os.path.join(output, f'{make_valid_filename(title)}.{format}')
        plt.savefig(file_path, format=format)
    else:
        plt.show()
    plt.close()


def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dir(
        output: str | None,
        data: bool = False,
        reports: bool = False,
        background: bool = False,
        classification: bool = False,
        regression: bool = False,
        batch: bool = False,
        pathways: bool = False,
    ) -> str | None:
    if not output:
        return None
    
    if data:
        title = 'preprocessed_data'
    elif reports:
        title = 'reports'
    elif background:
        title = 'background_scores'
    elif classification:
        title = 'cell_type_classification'
    elif regression:
        title = 'pseudotime_regression'
    else:
        title = ''
    
    path = os.path.join(output, title)
    if pathways:
        path = os.path.join(path, 'pathways')
    elif batch:
        path = os.path.join(path, 'batches')

    create_dir(path)
    return path
