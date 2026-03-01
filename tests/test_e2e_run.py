import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', PendingDeprecationWarning)

import matplotlib
matplotlib.use('Agg')

import argparse
import tempfile
import unittest
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from tests.interface import Test
from run import run_tool
from scripts.output import read_args, save_args, read_results, save_csv, convert_from_str
from scripts.consts import (
    CLASSIFICATION_METRIC, REGRESSION_METRIC,
    FEATURE_SELECTION, SET_FRACTION, DISTRIBUTIONS,
    SIZES, SEED, CELL_TYPE_COL, BackgroundMode
)
from scripts.backgrounds import define_sizes_in_real_mode as original_define_sizes_in_real_mode


TEST_RESULTS_DIR = os.path.join(os.getcwd(), 'tests', 'test_e2e_results')

class LogArgsTest(Test):
    def test_save_then_load_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                expression='expr.csv',
                cell_types=123,
                background_mode=BackgroundMode.RANDOM,
            )
            save_args(args, tmp)
            loaded = read_args(tmp)
            self.assertEqual(loaded, vars(args))


class E2ERunTest(Test):

    def setUp(self) -> None:
        np.random.seed(42)

        self.test_dir = os.path.join(os.getcwd(), 'tmp_e2e_test')
        input_dir = os.path.join(self.test_dir, 'input')

        for dir_path in [self.test_dir, input_dir, TEST_RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.expression = os.path.join(input_dir, 'expression.csv')
        self.cell_types = os.path.join(input_dir, 'cell_types.csv')
        self.pseudotime = os.path.join(input_dir, 'pseudotime.csv')
        self.reduction = os.path.join(input_dir, 'reduction.csv')
        self.custom_pathways = os.path.join(input_dir, 'custom_pathways.csv')
        
        # Expression
        expression = self.generate_data(num_genes=100, num_cells=50)
        expression.to_csv(self.expression)

        # Cell types
        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA'] * 25 + ['TypeB'] * 25
        }, index=expression.index)
        cell_types.to_csv(self.cell_types)

        # Pseudotime
        pseudotime_values_1 = np.random.uniform(0, 1, 50)
        pseudotime_values_2 = np.random.uniform(0, 1, 50)
        pseudotime_values_1[45:] = np.nan
        pseudotime_values_2[:5] = np.nan
        pseudotime = pd.DataFrame({
            'Lineage1': pseudotime_values_1,
            'Lineage2': pseudotime_values_2
        }, index=expression.index)
        pseudotime.to_csv(self.pseudotime)
        
        # Reduction
        reduction = pd.DataFrame({
            'UMAP1': np.random.normal(0, 1, 50),
            'UMAP2': np.random.normal(0, 1, 50)
        }, index=expression.index)
        reduction.to_csv(self.reduction)

        # Pathways
        custom_pathways = pd.DataFrame({
            'Pathway1': ['Gene3', 'Gene1', 'Gene2', '',  ''],
            'Pathway2': ['Gene10', 'Gene11', 'Gene12', 'Gene13', 'Gene14'],
            'Pathway3': ['Gene20', 'Gene21', '', '', ''],
        })
        custom_pathways.to_csv(self.custom_pathways, index=False)

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_args(self, background_mode: BackgroundMode, output: str, processes: int = 0):
        return {
            'expression': self.expression,
            'cell_types': self.cell_types,
            'pseudotime': self.pseudotime,
            'reduction': self.reduction,
            'preprocessed': True,
            'exclude_cell_types': [],
            'exclude_lineages': [],
            'organism': 'human',
            'pathway_database': [],
            'custom_pathways': [self.custom_pathways],
            'feature_selection': FEATURE_SELECTION,
            'set_fraction': SET_FRACTION,
            'min_set_size': SIZES[0],
            'classification_metric': CLASSIFICATION_METRIC,
            'regression_metric': REGRESSION_METRIC,
            'cross_validation': 2,
            'background_mode': background_mode,
            'random_sizes': SIZES,
            'repeats': 2,
            'effect_size_threshold': None,
            'corrected_effect_size': True,
            'seed': SEED,
            'distribution': DISTRIBUTIONS[0],
            'processes': processes,
            'mem': None,
            'time': None,
            'output': output,
            'cache': os.path.join(output, 'cache'),
            'tmp': os.path.join(output, 'tmp'),
            'verbose': False,
        }
    
    def _run_e2e(self, background_mode: BackgroundMode):
        output = os.path.join(self.test_dir, f'output_{background_mode.name.lower()}')
        for dir in [output, os.path.join(output, 'cache'), os.path.join(output, 'tmp')]:
            os.makedirs(dir, exist_ok=True)

        args = self.get_args(background_mode=background_mode, output=output)
        save_args(argparse.Namespace(**args), args['output'])
        run_tool(**args)

        output_files = [
            'expression.csv',
            'cell_types.csv', 
            'pseudotime.csv',
            'reduction.csv',
            'gene_sets.csv',
            f'{background_mode.name.lower()}_background_sizes.json',
            'cell_type_classification.csv',
            'pseudotime_regression.csv',
            'p_values_celltypes.csv',
            'p_values_pseudotime.csv',
        ]

        for file_name in output_files:
            path = os.path.join(output, file_name)
            self.assertTrue(os.path.exists(path), msg=f"{file_name} is missing")
            if file_name.endswith('.csv'):
                self.assertTrue(not pd.read_csv(path).empty, msg=f"{file_name} is empty")
            elif file_name.endswith('.json'):
                self.assertTrue(os.path.getsize(path) > 0, msg=f"{file_name} is empty")

        for dir in ['cell_types', 'pseudotime']:
            path = os.path.join(output, 'pathways', dir)
            self.assertTrue(os.path.exists(path), msg=f"{dir} directory is missing")
            self.assertTrue(len(os.listdir(path)) > 0, msg=f"{dir} directory is empty")
        
        classification: pd.DataFrame = read_results('cell_type_classification', output)
        regression: pd.DataFrame = read_results('pseudotime_regression', output)

        if not os.path.exists(os.path.join(TEST_RESULTS_DIR, f'classification_{background_mode.name.lower()}.csv')):
            save_csv(classification, f'classification_{background_mode.name.lower()}', TEST_RESULTS_DIR, keep_index=False)
        if not os.path.exists(os.path.join(TEST_RESULTS_DIR, f'regression_{background_mode.name.lower()}.csv')):
            save_csv(regression, f'regression_{background_mode.name.lower()}', TEST_RESULTS_DIR, keep_index=False)

        original_classification: pd.DataFrame = read_results(f'classification_{background_mode.name.lower()}', TEST_RESULTS_DIR)
        original_regression: pd.DataFrame = read_results(f'regression_{background_mode.name.lower()}', TEST_RESULTS_DIR)

        pd.testing.assert_frame_equal(classification.drop(['gene_importances'], axis=1), original_classification.drop(['gene_importances'], axis=1), atol=1e-3)  # type: ignore[union-attr]
        pd.testing.assert_frame_equal(regression.drop(['gene_importances'], axis=1), original_regression.drop(['gene_importances'], axis=1), atol=1e-3)  # type: ignore[union-attr]

        for i in range(len(classification)):
            original_importances = convert_from_str(original_classification.iloc[i]['gene_importances'])
            new_importances = convert_from_str(classification.iloc[i]['gene_importances'])
            for original, new in zip(original_importances, new_importances):  # type: ignore[arg-type]
                self.assertAlmostEqual(original, new, places=5)
        for i in range(len(regression)):
            original_importances = convert_from_str(original_regression.iloc[i]['gene_importances'])
            new_importances = convert_from_str(regression.iloc[i]['gene_importances'])
            for original, new in zip(original_importances, new_importances):  # type: ignore[arg-type]
                self.assertAlmostEqual(original, new, places=5)
            
    def test_e2e_local_run_random_mode(self):
        self._run_e2e(BackgroundMode.RANDOM)

    @patch('scripts.backgrounds.define_sizes_in_real_mode')
    def test_e2e_local_run_real_mode(self, mock_define_sizes):
        def side_effect(gene_sets, set_fraction, min_set_size, repeats):
            return original_define_sizes_in_real_mode(gene_sets, set_fraction, min_set_size, repeats=2)
        mock_define_sizes.side_effect = side_effect
        self._run_e2e(BackgroundMode.REAL)
          

if __name__ == '__main__':
    unittest.main()
