"""
Conversion script for AnnData (.h5ad) objects to Phoenix input format.

Exports expression matrix, cell-type annotations, pseudotime values,
and dimensionality reduction coordinates as CSV files compatible with Phoenix.

Usage (from the repository root):
    python -m converters.convert \
        --input my_data.h5ad \
        --output phoenix_input/ \
        --cell_type_key cell_type \
        --reduction_key X_umap

Run `python -m converters.convert --help` for all options.
"""

import argparse
import os

import anndata as ad
import pandas as pd
from scipy.sparse import issparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert an AnnData (.h5ad) object to Phoenix input CSV files.'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the AnnData .h5ad file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for the CSV files')

    # Expression
    parser.add_argument('--layer', type=str, default=None,
                        help='AnnData layer to use for expression (e.g. "counts", "raw_counts"). '
                             'If not specified, uses adata.raw.X if available, otherwise adata.X')
    parser.add_argument('--use_raw', action='store_true', default=False,
                        help='Use adata.raw for expression data (default: auto-detect)')

    # Cell types
    parser.add_argument('--cell_type_key', type=str, default=None,
                        help='Column name in adata.obs for cell-type annotations '
                             '(e.g. "cell_type", "leiden", "louvain")')

    # Pseudotime
    parser.add_argument('--pseudotime_key', type=str, nargs='*', default=None,
                        help='Column name(s) in adata.obs for pseudotime trajectories '
                             '(e.g. "dpt_pseudotime", "monocle3_pseudotime")')

    # Reduction
    parser.add_argument('--reduction_key', type=str, default=None,
                        help='Key in adata.obsm for dimensionality reduction coordinates '
                             '(e.g. "X_umap", "X_tsne", "X_pca"). Default: auto-detect')

    return parser.parse_args()


def get_expression(adata: ad.AnnData, layer: str | None, use_raw: bool, verbose: bool = True) -> pd.DataFrame:
    """Extract expression matrix as a DataFrame (cells x genes)."""
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
        genes = adata.var_names
        cells = adata.obs_names
        if verbose:
            print(f'Using expression from layer "{layer}"')
    elif use_raw or (adata.raw is not None and not use_raw):
        if adata.raw is None:
            raise ValueError("adata.raw is not available")
        X = adata.raw.X
        genes = adata.raw.var_names
        cells = adata.obs_names
        if verbose:
            print('Using expression from adata.raw.X')
    else:
        X = adata.X
        genes = adata.var_names
        cells = adata.obs_names
        if verbose:
            print('Using expression from adata.X')

    if issparse(X):
        X = X.toarray()

    return pd.DataFrame(X, index=cells, columns=genes)


def get_cell_types(adata: ad.AnnData, key: str, verbose: bool = True) -> pd.DataFrame:
    """Extract cell-type annotations as a single-column DataFrame."""
    if key not in adata.obs.columns:
        raise ValueError(
            f"Cell-type key '{key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    ct = adata.obs[[key]].copy()
    ct.columns = ['cell_type']
    if verbose:
        print(f'Extracted {ct["cell_type"].nunique()} cell types from obs["{key}"]')
    return ct


def get_pseudotime(adata: ad.AnnData, keys: list[str], verbose: bool = True) -> pd.DataFrame:
    """Extract pseudotime trajectories as a DataFrame."""
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Pseudotime key(s) {missing} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    pt = adata.obs[keys].copy().astype(float)
    if verbose:
        print(f'Extracted {len(keys)} pseudotime trajectory(ies): {", ".join(keys)}')
    return pt


def get_reduction(adata: ad.AnnData, key: str | None, verbose: bool = True) -> pd.DataFrame:
    """Extract dimensionality reduction coordinates (first 2 components)."""
    if key is not None:
        if key not in adata.obsm:
            raise ValueError(
                f"Reduction key '{key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
    else:
        # Auto-detect: prefer UMAP > t-SNE > PCA
        for candidate in ['X_umap', 'X_tsne', 'X_pca']:
            if candidate in adata.obsm:
                key = candidate
                break
        if key is None:
            if verbose:
                print('No dimensionality reduction found in adata.obsm — skipping. '
                      'Phoenix will compute one automatically using --reduction.')
            return None  # type: ignore[return-value]

    coords = adata.obsm[key][:, :2]
    method = key.replace('X_', '')
    columns = [f'{method}_1', f'{method}_2']
    if verbose:
        print(f'Extracted reduction from obsm["{key}"]')
    return pd.DataFrame(coords, index=adata.obs_names, columns=columns)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load AnnData
    print(f'Loading {args.input}...')
    adata = ad.read_h5ad(args.input)
    print(f'Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes')

    # Expression
    expression = get_expression(adata, args.layer, args.use_raw)
    expression.to_csv(os.path.join(args.output, 'expression.csv'))
    print(f'  -> Saved expression.csv ({expression.shape[0]} cells x {expression.shape[1]} genes)')

    # Cell types
    if args.cell_type_key:
        cell_types = get_cell_types(adata, args.cell_type_key)
        cell_types.to_csv(os.path.join(args.output, 'cell_types.csv'))
        print(f'  -> Saved cell_types.csv')

    # Pseudotime
    if args.pseudotime_key:
        pseudotime = get_pseudotime(adata, args.pseudotime_key)
        pseudotime.to_csv(os.path.join(args.output, 'pseudotime.csv'))
        print(f'  -> Saved pseudotime.csv')

    # Reduction
    reduction = get_reduction(adata, args.reduction_key)
    if reduction is not None:
        reduction.to_csv(os.path.join(args.output, 'reduction.csv'))
        print(f'  -> Saved reduction.csv')

    # Summary
    print(f'\nConversion complete. Output files in: {args.output}/')
    print('You can now run Phoenix:')
    cmd_parts = [
        f'python run.py',
        f'    --expression {args.output}/expression.csv',
    ]
    if args.cell_type_key:
        cmd_parts.append(f'    --cell_types {args.output}/cell_types.csv')
    if args.pseudotime_key:
        cmd_parts.append(f'    --pseudotime {args.output}/pseudotime.csv')
    if reduction is not None:
        cmd_parts.append(f'    --reduction {args.output}/reduction.csv')
    cmd_parts.append(f'    --output <output_dir>')
    print('  ' + ' \\\n  '.join(cmd_parts))


if __name__ == '__main__':
    main()
