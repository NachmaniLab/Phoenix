import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', PendingDeprecationWarning)

import unittest
import shutil
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

from tests.interface import Test
from run import run_tool
from scripts.consts import (
    CLASSIFIER, REGRESSOR, CLASSIFICATION_METRIC, REGRESSION_METRIC,
    FEATURE_SELECTION, SET_FRACTION, DISTRIBUTIONS,
    MIN_SET_SIZE, SEED, CELL_TYPE_COL
)
from scripts.data import preprocess_data as real_preprocess_data


class E2ERunTest(Test):

    def setUp(self) -> None:
        self.test_dir = os.path.join(os.getcwd(), 'tmp_e2e_test')
        self.output = os.path.join(self.test_dir, 'output')
        self.cache = os.path.join(self.test_dir, 'cache')
        self.tmp = os.path.join(self.test_dir, 'tmp')
        input_dir = os.path.join(self.test_dir, 'input')

        for dir_path in [self.test_dir, self.output, self.cache, self.tmp, input_dir]:
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

    def get_args(self, processes: int = 0):
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
            'min_set_size': MIN_SET_SIZE,
            'classifier': CLASSIFIER,
            'regressor': REGRESSOR,
            'classification_metric': CLASSIFICATION_METRIC,
            'regression_metric': REGRESSION_METRIC,
            'cross_validation': 2,
            'repeats': 2,
            'seed': SEED,
            'distribution': DISTRIBUTIONS[0],
            'processes': processes,
            'mem': None,
            'time': None,
            'output': self.output,
            'cache': self.cache,
            'tmp': self.tmp,
            'verbose': False,
        }
    
    def test_e2e_local_run(self):

        run_tool(**self.get_args())

        output_files = [
            'expression.csv',
            'cell_types.csv', 
            'pseudotime.csv',
            'reduction.csv',
            'gene_sets.csv',
            'cell_type_classification.csv',
            'pseudotime_regression.csv',
            'p_values_celltypes.csv',
            'p_values_pseudotime.csv',
        ]
        for file_name in output_files:
            path = os.path.join(self.output, file_name)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(not pd.read_csv(path).empty)

        for dir in ['cell_types', 'pseudotime']:
            path = os.path.join(self.output, 'pathways', dir)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(len(os.listdir(path)) > 0)


if __name__ == '__main__':
    unittest.main()
