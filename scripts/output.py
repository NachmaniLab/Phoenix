import argparse
from enum import Enum
import os, yaml, glob, json, gzip  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scripts.consts import TARGET_COL, CELL_TYPE_COL, BackgroundMode
from scripts.utils import enum2str, make_valid_filename, convert_from_str


# TODO: add methods to return paths of cache, reports and batch results (if needed)


def _get_args_filename() -> str:
    return 'run_args.json'


def save_args(args: argparse.Namespace, output_path: str) -> None:
    for key, value in vars(args).items():
        if isinstance(value, Enum):
            setattr(args, key, enum2str(value))
    path = os.path.join(output_path, _get_args_filename())
    with open(path, 'w') as file:
        json.dump(vars(args), file, indent=4)


def read_args(output_path: str) -> dict:
    path = os.path.join(output_path, _get_args_filename())
    with open(path, 'r') as file:
        return json.load(file)


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
     
    if data is None or (hasattr(data, 'empty') and data.empty):
        return
    if isinstance(data, list):
        if not data:
            return

    if isinstance(data, dd.DataFrame):
        data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), single_file=True, index=keep_index)

    else:
        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), index=keep_index)


def _read_10x_mtx(mtx_dir: str) -> pd.DataFrame:
    """
    Read 10x Genomics MTX folder format and return a pandas DataFrame.
    
    Expected files in mtx_dir:
    - matrix.mtx or matrix.mtx.gz: Matrix Market format (genes x cells)
    - features.tsv or features.tsv.gz (or genes.tsv(.gz)): Gene names
    - barcodes.tsv or barcodes.tsv.gz: Cell barcodes
    
    Returns DataFrame with genes as columns and cells as rows.
    """
    # Find matrix file
    matrix_file = None
    for fname in ['matrix.mtx.gz', 'matrix.mtx']:
        fpath = os.path.join(mtx_dir, fname)
        if os.path.exists(fpath):
            matrix_file = fpath
            break
    
    if matrix_file is None:
        raise FileNotFoundError(f"No matrix.mtx or matrix.mtx.gz found in {mtx_dir}")
    
    # Find features file
    features_file = None
    for fname in ['features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv']:
        fpath = os.path.join(mtx_dir, fname)
        if os.path.exists(fpath):
            features_file = fpath
            break
    
    if features_file is None:
        raise FileNotFoundError(f"No features.tsv or genes.tsv found in {mtx_dir}")
    
    # Find barcodes file
    barcodes_file = None
    for fname in ['barcodes.tsv.gz', 'barcodes.tsv']:
        fpath = os.path.join(mtx_dir, fname)
        if os.path.exists(fpath):
            barcodes_file = fpath
            break
    
    if barcodes_file is None:
        raise FileNotFoundError(f"No barcodes.tsv or barcodes.tsv.gz found in {mtx_dir}")
    
    # Read matrix
    if matrix_file.endswith('.gz'):
        with gzip.open(matrix_file, 'rb') as f:
            mtx = mmread(f)
    else:
        mtx = mmread(matrix_file)
    
    # Convert to CSR format for efficiency
    mtx = csr_matrix(mtx)
    
    # Read features (genes)
    if features_file.endswith('.gz'):
        with gzip.open(features_file, 'rt') as f:
            features_df = pd.read_csv(f, sep='\t', header=None)
    else:
        features_df = pd.read_csv(features_file, sep='\t', header=None)
    
    # Use gene_name column (column 1) if available, otherwise use first column
    if features_df.shape[1] >= 2:
        genes = features_df.iloc[:, 1].tolist()
    else:
        genes = features_df.iloc[:, 0].tolist()
    
    # Read barcodes
    if barcodes_file.endswith('.gz'):
        with gzip.open(barcodes_file, 'rt') as f:
            barcodes = pd.read_csv(f, sep='\t', header=None).iloc[:, 0].tolist()
    else:
        barcodes = pd.read_csv(barcodes_file, sep='\t', header=None).iloc[:, 0].tolist()
    
    # Validate dimensions
    if mtx.shape[0] != len(genes):
        raise ValueError(f"Matrix rows ({mtx.shape[0]}) != number of genes ({len(genes)})")
    if mtx.shape[1] != len(barcodes):
        raise ValueError(f"Matrix columns ({mtx.shape[1]}) != number of barcodes ({len(barcodes)})")
    
    # Create DataFrame: transpose so cells are rows and genes are columns
    # Convert directly to dense for compatibility with existing pipeline
    return pd.DataFrame(mtx.T.toarray(), index=barcodes, columns=genes)


def _read_expression(expression_path: str) -> pd.DataFrame:
    """
    Read expression data from either CSV file or 10x MTX directory.
    
    Args:
        expression_path: Path to CSV file or 10x MTX directory
        
    Returns:
        DataFrame with cells as rows and genes as columns
    """
    if os.path.isfile(expression_path):
        # CSV file
        if expression_path.endswith('.csv'):
            return read_csv(expression_path)
        else:
            raise ValueError(f"File '{expression_path}' must be a CSV file")
    elif os.path.isdir(expression_path):
        # Check if it's a 10x MTX directory
        has_mtx = any(
            os.path.exists(os.path.join(expression_path, f)) 
            for f in ['matrix.mtx', 'matrix.mtx.gz']
        )
        if has_mtx:
            return _read_10x_mtx(expression_path)
        else:
            raise ValueError(
                f"Directory '{expression_path}' does not contain required 10x MTX files. "
                "Expected: matrix.mtx(.gz), features.tsv(.gz), barcodes.tsv(.gz)"
            )
    else:
        raise FileNotFoundError(f"Expression path '{expression_path}' does not exist")


def read_raw_data(expression: str, cell_types: str | None, pseudotime: str | None, reduction: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression = _read_expression(expression)
    cell_types = read_csv(cell_types).loc[expression.index] if cell_types else None
    if cell_types is not None:
        cell_types = cell_types.rename(columns={cell_types.columns[0]: CELL_TYPE_COL})
    pseudotime = read_csv(pseudotime).loc[expression.index] if pseudotime else None 
    if os.path.exists(reduction):
        reduction = read_csv(reduction).loc[expression.index]
    return expression, cell_types, pseudotime, reduction


def _get_size_filename(background_mode: BackgroundMode) -> str:
    return make_valid_filename(f'{background_mode.name}_background_sizes.json')


def save_sizes(sizes: list[int], background_mode: BackgroundMode, output_path: str) -> None:
    path = os.path.join(output_path, _get_size_filename(background_mode))
    with open(path, 'w') as file:
        json.dump(sizes, file)


def load_sizes(output_path: str) -> tuple[list[int], BackgroundMode]:
    for background_mode in [BackgroundMode.REAL, BackgroundMode.RANDOM]:
        path = os.path.join(output_path, _get_size_filename(background_mode))
        if os.path.exists(path):
            with open(path, 'r') as file:
                sizes = json.load(file)
            return sizes, background_mode
    raise FileNotFoundError('No background sizes found in output path.')
    

def load_background_scores(background: str, cache_path: str | None = None, verbose: bool = False):
    background = make_valid_filename(background).lower()
    if cache_path and os.path.exists(f'{cache_path}/{background}.yml') and os.path.getsize(f'{cache_path}/{background}.yml') > 0:
        if verbose:
            print(f'Loading background {background} from cache...')
        with open(f'{cache_path}/{background}.yml', 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    raise FileNotFoundError(f'Background scores for {background} are missing or empty in cache at {cache_path}')


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


def read_results(title: str, output_path: str, index_col: int | None = None, raise_err: bool = False) -> pd.DataFrame | None:
    try:
        title = f'{title}.csv' if '.csv' not in title else title
        return read_csv(os.path.join(output_path, f'{make_valid_filename(title)}'), index_col=index_col)  # type: ignore[arg-type]
    except Exception as e:
        if raise_err:
            raise e
        return None


def get_preprocessed_data(data: pd.DataFrame | str, output_path: str) -> pd.DataFrame:
    if isinstance(data, str):
        data = read_results(data, output_path, index_col=0)
    return data


def aggregate_batch_results(tmp: str, result_type: str) -> pd.DataFrame | None:
    dfs = []
    for path in glob.glob(os.path.join(tmp, f'{result_type}_batch*.csv')):  # type: ignore[arg-type]
        df = read_results(os.path.basename(path), tmp, index_col=None)  # type: ignore[arg-type]
        if df is not None:
            dfs.append(df)
    if not dfs:  # if result type is missing
        return None
    return pd.concat(dfs, ignore_index=True)


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
