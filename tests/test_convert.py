import os
import shutil
import subprocess
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from tests.interface import Test
from converters.convert import get_expression, get_cell_types, get_pseudotime, get_reduction
from scripts.output import read_raw_data
from scripts.data import preprocess_data
from scripts.consts import SEED


def _r_available() -> bool:
    """Check if Rscript and Seurat are available."""
    try:
        result = subprocess.run(
            ['Rscript', '-e', 'library(Seurat); library(optparse)'],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


R_AVAILABLE = _r_available()


class TestAnnDataConversion(Test):

    def setUp(self) -> None:
        np.random.seed(42)
        n_cells, n_genes = 30, 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells),
            'pseudotime': np.random.rand(n_cells),
        }, index=[f'Cell{i}' for i in range(n_cells)])
        var = pd.DataFrame(index=[f'Gene{i}' for i in range(n_genes)])

        self.adata = ad.AnnData(X=X, obs=obs, var=var)
        self.adata.layers['counts'] = np.random.randint(0, 100, (n_cells, n_genes)).astype(np.float32)
        self.adata.obsm['X_umap'] = np.random.rand(n_cells, 2).astype(np.float32)
        self.adata.obsm['X_pca'] = np.random.rand(n_cells, 10).astype(np.float32)

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Expression ---

    def test_expression_from_X(self) -> None:
        expr = get_expression(self.adata, layer=None, use_raw=False, verbose=False)
        self.assertEqual(expr.shape, (30, 50))
        self.assertListEqual(list(expr.index), list(self.adata.obs_names))
        self.assertListEqual(list(expr.columns), list(self.adata.var_names))

    def test_expression_from_layer(self) -> None:
        expr = get_expression(self.adata, layer='counts', use_raw=False, verbose=False)
        np.testing.assert_array_equal(expr.values, self.adata.layers['counts'])

    def test_expression_missing_layer_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_expression(self.adata, layer='nonexistent', use_raw=False, verbose=False)

    def test_expression_sparse_matrix(self) -> None:
        self.adata.X = csr_matrix(self.adata.X)
        expr = get_expression(self.adata, layer=None, use_raw=False, verbose=False)
        self.assertEqual(expr.shape, (30, 50))
        self.assertFalse(hasattr(expr.values, 'toarray'))

    # --- Cell types ---

    def test_cell_types(self) -> None:
        ct = get_cell_types(self.adata, 'cell_type', verbose=False)
        self.assertEqual(list(ct.columns), ['cell_type'])
        self.assertEqual(len(ct), 30)
        self.assertTrue(set(ct['cell_type'].unique()).issubset({'TypeA', 'TypeB', 'TypeC'}))

    def test_cell_types_missing_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_cell_types(self.adata, 'nonexistent', verbose=False)

    # --- Pseudotime ---

    def test_pseudotime(self) -> None:
        pt = get_pseudotime(self.adata, ['pseudotime'], verbose=False)
        self.assertEqual(pt.shape, (30, 1))
        self.assertTrue(pt.dtypes.iloc[0] == np.float64)

    def test_pseudotime_missing_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_pseudotime(self.adata, ['nonexistent'], verbose=False)

    # --- Reduction ---

    def test_reduction_explicit_key(self) -> None:
        red = get_reduction(self.adata, 'X_umap', verbose=False)
        self.assertEqual(red.shape, (30, 2))
        self.assertListEqual(list(red.columns), ['umap_1', 'umap_2'])

    def test_reduction_auto_detect(self) -> None:
        red = get_reduction(self.adata, key=None, verbose=False)
        self.assertIsNotNone(red)
        self.assertEqual(red.shape, (30, 2))
        self.assertIn('umap', red.columns[0])

    def test_reduction_missing_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_reduction(self.adata, 'X_nonexistent', verbose=False)

    def test_reduction_none_when_empty(self) -> None:
        self.adata.obsm = {}
        red = get_reduction(self.adata, key=None, verbose=False)
        self.assertIsNone(red)

    # --- CSV round-trip ---

    def test_csv_output_is_phoenix_compatible(self) -> None:
        """Verify exported CSVs match the format Phoenix expects."""
        expr = get_expression(self.adata, layer=None, use_raw=False, verbose=False)
        expr.to_csv(os.path.join(self.tmpdir, 'expression.csv'))

        ct = get_cell_types(self.adata, 'cell_type', verbose=False)
        ct.to_csv(os.path.join(self.tmpdir, 'cell_types.csv'))

        # Re-read and validate structure
        expr_read = pd.read_csv(os.path.join(self.tmpdir, 'expression.csv'), index_col=0)
        self.assertEqual(expr_read.shape, (30, 50))

        ct_read = pd.read_csv(os.path.join(self.tmpdir, 'cell_types.csv'), index_col=0)
        self.assertEqual(ct_read.shape[1], 1)
        self.assertEqual(list(ct_read.columns), ['cell_type'])

    # --- Integration with Phoenix pipeline ---

    def test_converted_files_accepted_by_phoenix_setup(self) -> None:
        """Run Phoenix read_raw_data + preprocess_data on converted CSVs to confirm compatibility."""
        # Export all files from AnnData
        expr = get_expression(self.adata, layer=None, use_raw=False, verbose=False)
        expr.to_csv(os.path.join(self.tmpdir, 'expression.csv'))

        ct = get_cell_types(self.adata, 'cell_type', verbose=False)
        ct.to_csv(os.path.join(self.tmpdir, 'cell_types.csv'))

        pt = get_pseudotime(self.adata, ['pseudotime'], verbose=False)
        pt.to_csv(os.path.join(self.tmpdir, 'pseudotime.csv'))

        red = get_reduction(self.adata, 'X_umap', verbose=False)
        red.to_csv(os.path.join(self.tmpdir, 'reduction.csv'))

        # Read back through Phoenix's own reader
        expression, cell_types, pseudotime, reduction = read_raw_data(
            expression=os.path.join(self.tmpdir, 'expression.csv'),
            cell_types=os.path.join(self.tmpdir, 'cell_types.csv'),
            pseudotime=os.path.join(self.tmpdir, 'pseudotime.csv'),
            reduction=os.path.join(self.tmpdir, 'reduction.csv'),
        )

        self.assertEqual(expression.shape, (30, 50))
        self.assertEqual(cell_types.shape, (30, 1))
        self.assertEqual(pseudotime.shape, (30, 1))
        self.assertEqual(reduction.shape, (30, 2))

        # Run Phoenix preprocessing (step 1 core logic)
        output_dir = os.path.join(self.tmpdir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        expression, cell_types, pseudotime, reduction = preprocess_data(
            expression, cell_types, pseudotime, reduction,
            preprocessed=True, seed=SEED, output=output_dir, verbose=False,
        )

        self.assertGreater(len(expression), 0)
        self.assertGreater(len(cell_types), 0)
        self.assertGreater(len(pseudotime), 0)
        self.assertGreater(len(reduction), 0)

        # Verify preprocessed CSVs were saved
        for fname in ['expression.csv', 'cell_types.csv', 'pseudotime.csv', 'reduction.csv']:
            self.assertTrue(os.path.exists(os.path.join(output_dir, fname)))


# R script that creates a minimal Seurat .rds object for testing
_CREATE_SEURAT_R = r"""
suppressPackageStartupMessages(library(Seurat))

set.seed(42)
n_cells <- 30
n_genes <- 50

counts <- matrix(
    rpois(n_cells * n_genes, lambda = 5),
    nrow = n_genes, ncol = n_cells,
    dimnames = list(paste0("Gene", 0:(n_genes - 1)), paste0("Cell", 0:(n_cells - 1)))
)

obj <- CreateSeuratObject(counts = counts)
obj[["cell_type"]] <- rep(c("TypeA", "TypeB", "TypeC"), length.out = n_cells)
obj[["pseudotime"]] <- runif(n_cells)

obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = 10, verbose = FALSE)

saveRDS(obj, file = commandArgs(trailingOnly = TRUE)[1])
cat("Saved test Seurat object\n")
"""

CONVERT_R_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'converters', 'convert.R')


@unittest.skipUnless(R_AVAILABLE, 'R with Seurat/optparse not available')
class TestSeuratConversion(Test):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.rds_path = os.path.join(self.tmpdir, 'test.rds')
        self.out_dir = os.path.join(self.tmpdir, 'output')
        os.makedirs(self.out_dir, exist_ok=True)

        # Create Seurat object via R
        r_script = os.path.join(self.tmpdir, 'create_seurat.R')
        with open(r_script, 'w') as f:
            f.write(_CREATE_SEURAT_R)
        result = subprocess.run(
            ['Rscript', r_script, self.rds_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            self.fail(f'Failed to create Seurat object:\n{result.stderr}')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_convert(self, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
        cmd = [
            'Rscript', CONVERT_R_PATH,
            '--input', self.rds_path,
            '--output', self.out_dir,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # --- Expression ---

    def test_expression_output(self) -> None:
        result = self._run_convert()
        self.assertEqual(result.returncode, 0, result.stderr)
        expr = pd.read_csv(os.path.join(self.out_dir, 'expression.csv'), index_col=0)
        self.assertEqual(expr.shape, (30, 50))
        self.assertTrue(all(c.startswith('Gene') for c in expr.columns))
        self.assertTrue(all(r.startswith('Cell') for r in expr.index))

    def test_expression_data_slot(self) -> None:
        result = self._run_convert(['--slot', 'data'])
        self.assertEqual(result.returncode, 0, result.stderr)
        expr = pd.read_csv(os.path.join(self.out_dir, 'expression.csv'), index_col=0)
        self.assertEqual(expr.shape, (30, 50))

    # --- Cell types ---

    def test_cell_types_output(self) -> None:
        result = self._run_convert(['--cell_type_key', 'cell_type'])
        self.assertEqual(result.returncode, 0, result.stderr)
        ct = pd.read_csv(os.path.join(self.out_dir, 'cell_types.csv'), index_col=0)
        self.assertEqual(list(ct.columns), ['cell_type'])
        self.assertEqual(len(ct), 30)
        self.assertTrue(set(ct['cell_type'].unique()).issubset({'TypeA', 'TypeB', 'TypeC'}))

    def test_cell_types_missing_key_fails(self) -> None:
        result = self._run_convert(['--cell_type_key', 'nonexistent'])
        self.assertNotEqual(result.returncode, 0)

    # --- Pseudotime ---

    def test_pseudotime_output(self) -> None:
        result = self._run_convert(['--pseudotime_key', 'pseudotime'])
        self.assertEqual(result.returncode, 0, result.stderr)
        pt = pd.read_csv(os.path.join(self.out_dir, 'pseudotime.csv'), index_col=0)
        self.assertEqual(pt.shape, (30, 1))

    def test_pseudotime_missing_key_fails(self) -> None:
        result = self._run_convert(['--pseudotime_key', 'nonexistent'])
        self.assertNotEqual(result.returncode, 0)

    # --- Reduction ---

    def test_reduction_auto_detect(self) -> None:
        result = self._run_convert()
        self.assertEqual(result.returncode, 0, result.stderr)
        red = pd.read_csv(os.path.join(self.out_dir, 'reduction.csv'), index_col=0)
        self.assertEqual(red.shape, (30, 2))

    def test_reduction_explicit_key(self) -> None:
        result = self._run_convert(['--reduction_key', 'pca'])
        self.assertEqual(result.returncode, 0, result.stderr)
        red = pd.read_csv(os.path.join(self.out_dir, 'reduction.csv'), index_col=0)
        self.assertEqual(red.shape, (30, 2))
        self.assertIn('pca', red.columns[0])

    def test_reduction_missing_key_fails(self) -> None:
        result = self._run_convert(['--reduction_key', 'nonexistent'])
        self.assertNotEqual(result.returncode, 0)

    # --- Integration with Phoenix pipeline ---

    def test_converted_files_accepted_by_phoenix_setup(self) -> None:
        """Run Phoenix read_raw_data + preprocess_data on Seurat-converted CSVs."""
        result = self._run_convert([
            '--cell_type_key', 'cell_type',
            '--pseudotime_key', 'pseudotime',
            '--reduction_key', 'pca',
        ])
        self.assertEqual(result.returncode, 0, result.stderr)

        expression, cell_types, pseudotime, reduction = read_raw_data(
            expression=os.path.join(self.out_dir, 'expression.csv'),
            cell_types=os.path.join(self.out_dir, 'cell_types.csv'),
            pseudotime=os.path.join(self.out_dir, 'pseudotime.csv'),
            reduction=os.path.join(self.out_dir, 'reduction.csv'),
        )

        self.assertEqual(expression.shape, (30, 50))
        self.assertEqual(cell_types.shape, (30, 1))
        self.assertEqual(pseudotime.shape, (30, 1))
        self.assertEqual(reduction.shape, (30, 2))

        output_dir = os.path.join(self.tmpdir, 'phoenix_output')
        os.makedirs(output_dir, exist_ok=True)

        expression, cell_types, pseudotime, reduction = preprocess_data(
            expression, cell_types, pseudotime, reduction,
            preprocessed=True, seed=SEED, output=output_dir, verbose=False,
        )

        self.assertGreater(len(expression), 0)
        self.assertGreater(len(cell_types), 0)
        self.assertGreater(len(pseudotime), 0)
        self.assertGreater(len(reduction), 0)
