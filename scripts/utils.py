import re, os, time
import pandas as pd
from functools import wraps
from argparse import Namespace
import numpy as np
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scripts.consts import SIZES, LIST_SEP, CELL_TYPE_COL, ALL_CELLS


transform_log = lambda x: np.log2(x + 1)
re_transform_log = lambda x: 2 ** x - 1
adjust_p_value = lambda p_values: multipletests(p_values, method='fdr_bh')[1]


def get_full_path(path: str) -> str:
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    return os.path.abspath(path)


def get_file_size(path: str) -> str:
    """
    return: rounded file size in GB or MB
    """
    file_size_bytes = os.path.getsize(path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    
    if file_size_gb < 1:
        file_size_mb = file_size_bytes / (1024 ** 2)
        return f"{int(np.ceil(file_size_mb))}M"
    else:
        return f"{int(np.ceil(file_size_gb))}G"
    

def make_valid_filename(filename: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.]+', '', filename.replace(' ', '_')).lower()[:100]


def make_valid_term(term: str) -> str:
    return term.replace(',', '')


def convert_to_sci(num: float) -> str:
    return '{:.0e}'.format(num)


def convert_to_str(info: str | list | dict | None) -> str:
    if isinstance(info, list):
        return LIST_SEP.join([convert_to_str(i) for i in info])
    if isinstance(info, dict):
        return LIST_SEP.join(f'{convert_to_str(k)}: {convert_to_str(v)}' for k, v in info.items())
    return str(info)


def convert_from_str(info: str) -> list | float | str:
    if isinstance(info, str) and LIST_SEP in info:
        return [convert_from_str(i) for i in info.split(LIST_SEP)]
    try:
        return float(info)
    except:
        return info


def define_task(cell_type: str | None = None, lineage: str | None = None):
    if lineage:
        return f'regression_{lineage}'
    if cell_type:
        return f'classification_{cell_type}'
    raise RuntimeError()


def define_background(set_size: int, repeats: int, cell_type: str | None = None, lineage: str | None = None):
    # TODO: add info such as model name
    return f'{define_task(cell_type, lineage)}_size{set_size}_repeats{repeats}'


def define_set_size(set_len: int, set_fraction: float, min_set_size: int) -> int:
    set_size = min(max(int(set_len * set_fraction), min_set_size), set_len)
    return max((x for x in SIZES if x <= set_size), default=None)  # type: ignore[arg-type]


def define_batch_size(gene_set_len: int, processes: int) -> int:
    if not processes:
        return gene_set_len
    return int(np.ceil(gene_set_len / processes))


def define_task_length(cell_types, pseudotime) -> int:
    return len(cell_types[CELL_TYPE_COL].unique()) + pseudotime.shape[1]


def parse_missing_args(args):
    args_dict = vars(args)
    updated_args = {k: (None if v == 'None' else v) for k, v in args_dict.items()}
    return Namespace(**updated_args)


def get_color_mapping(cell_types: list[str]) -> dict[str, str]:
    """
    cell_types: unique
    """
    color_palette = sns.color_palette('Paired', n_colors=len(cell_types))
    return {cell_type: color_palette[i] for i, cell_type in enumerate(cell_types)}


def remove_outliers(values: list[float], k: float = 1.5) -> list[float]:
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return [i for i in values if lower <= i <= upper]



def correct_effect_size(effect_sizes: pd.Series, targets: pd.Series) -> pd.Series:
    corrected_effect_sizes = effect_sizes.copy()
    for target in targets.unique():
        if target == ALL_CELLS:
            continue
        corrected_effect_sizes[targets == target] -= effect_sizes[targets == target].mean()
    return corrected_effect_sizes


def show_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)        
        if seconds > 5:
            info = f" on data with shape {kwargs.get('X').shape} using {kwargs.get('predictor').__name__}" if kwargs.get('X') is not None and kwargs.get('predictor') is not None else ""
            print(f'Running {func.__name__}{info} took {f"{minutes:02d}:{seconds:02d}"} minutes to run.')
        return result
    return wrapper
