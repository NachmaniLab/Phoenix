import numpy as np
from scripts.consts import SIZES, REPEATS, BackgroundMode
from scripts.utils import define_set_size
from scripts.output import save_sizes


def set_background_mode(background_mode: BackgroundMode, gene_set_len: int) -> BackgroundMode:
    if background_mode == BackgroundMode.AUTO:
        # By definition, we use our defined REPEATS and SIZES and not provided args
        return BackgroundMode.REAL if gene_set_len >= REPEATS * len(SIZES) else BackgroundMode.RANDOM
    return background_mode


def define_sizes_in_random_mode(gene_sets: dict[str, list[str]], set_fraction: float, min_set_size: int, random_sizes: list[int]) -> list[int]:
    sizes_used: set[int] = set()
    for genes in gene_sets.values():
        size = define_set_size(len(genes), set_fraction, min_set_size, all_sizes=random_sizes)
        sizes_used.add(size)
    return sorted(sizes_used)


def define_sizes_in_real_mode(gene_sets: dict[str, list[str]], set_fraction: float, min_set_size: int, repeats: int) -> list[int]:
    len_sizes = len(gene_sets) // repeats
    if len_sizes == 0:
        raise RuntimeError('Not enough gene sets for real background mode. Consider using random background mode instead.')
    
    # Compute final (post-fraction, post-min, post-cap) size per gene set
    sizes = []
    for genes in gene_sets.values():
        set_len = len(genes)
        target = int(set_len * set_fraction)
        target = max(target, min_set_size)
        target = min(target, set_len)
        sizes.append(target)
    sizes.sort()

    # Split into `len_sizes` bins and take median of each bin
    n = len(sizes)
    sizes_used: list[int] = []
    for b in range(len_sizes):
        start = (b * n) // len_sizes
        end = ((b + 1) * n) // len_sizes
        if start == end:  # if len_sizes > n
            continue
        bin_vals = sizes[start:end]
        sizes_used.append(int(np.median(bin_vals)))
    return sorted(set(sizes_used))


def define_sizes(background_mode: BackgroundMode, gene_sets: dict[str, list[str]], set_fraction: float, min_set_size: int, repeats: int, random_sizes: list[int], output: str) -> list[int]:
    if background_mode == BackgroundMode.RANDOM:
        sizes_used = define_sizes_in_random_mode(gene_sets, set_fraction, min_set_size, random_sizes)
    else:
        sizes_used = define_sizes_in_real_mode(gene_sets, set_fraction, min_set_size, repeats)
    save_sizes(sizes_used, background_mode, output)
    return sizes_used
