import tempfile
import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from scripts.backgrounds import define_sizes_in_real_mode
from tests.interface import Test
from scripts.calculate_scores import _get_target_size_pair_batch, calculate_background_scores_in_random_mode, calculate_pathway_scores, calculate_background_scores_in_real_mode
from scripts.consts import (
    CELL_TYPE_COL, TARGET_COL, FEATURE_SELECTION, MIN_SET_SIZE,
    CLASSIFIER, REGRESSOR, CLASSIFICATION_METRIC, REGRESSION_METRIC, SEED
)
from scripts.output import aggregate_batch_results


class CalculatePathwayScoresTest(Test):

    def setUp(self):
        expression = self.generate_data(num_cells=20, num_genes=30)
        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA'] * 10 + ['TypeB'] * 10,
        }, index=expression.index)
        self.pseudotime = pd.DataFrame({
            'Lineage1': np.linspace(0, 1, 20),
            'Lineage2': np.random.uniform(0, 1, 20),
        }, index=expression.index)
        self.gene_sets = {
            'Pathway1': [f'Gene{i}' for i in range(2)],
            'Pathway2': [f'Gene{i}' for i in range(10, 15)],
            'Pathway3': [f'Gene{i}' for i in range(20, 30)],
            'Pathway4': [f'Gene{i}' for i in range(5, 55)],
        }
        sizes = define_sizes_in_real_mode(self.gene_sets, 1.0, MIN_SET_SIZE, repeats=1)
        self.default_params = {
            'expression': expression,
            'cell_types': self.cell_types,
            'pseudotime': self.pseudotime,
            'gene_sets': self.gene_sets,
            'sizes': sizes,
            'feature_selection': FEATURE_SELECTION,
            'min_set_size': MIN_SET_SIZE,
            'classifier': CLASSIFIER,
            'regressor': REGRESSOR,
            'classification_metric': CLASSIFICATION_METRIC,
            'regression_metric': REGRESSION_METRIC,
            'seed': SEED,
            'set_fraction': 1.0,
            'cross_validation': 2,
            'processes': 0,
            'output': '',
            'tmp': '',
            'verbose': False,
        }

    def test_pathway_scores_happy_flow(self):
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            classification, regression = calculate_pathway_scores(**self.default_params)

        expected_columns = [
            TARGET_COL, 'set_name', 'top_genes', 'gene_importances', 
            'set_size', 'pathway_score', 'effect_size'
        ]
        self.assertTrue(all(col in classification.columns for col in expected_columns))
        self.assertTrue(all(col in regression.columns for col in expected_columns))

        self.assertEqual(len(classification), len(self.gene_sets) * (len(self.cell_types[CELL_TYPE_COL].unique()) + 1))  # plus 'All'
        self.assertEqual(len(regression), len(self.gene_sets) * len(self.pseudotime.columns))

        self.assertFalse(classification['pathway_score'].isnull().any())
        self.assertFalse(regression['pathway_score'].isnull().any())

    def test_without_cell_types(self):
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            args = self.default_params.copy()
            args['cell_types'] = None
            classification, regression = calculate_pathway_scores(**args)
        self.assertEqual(len(classification), 0)
        self.assertGreater(len(regression), 0)

    def test_without_pseudotime(self):
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            args = self.default_params.copy()
            args['pseudotime'] = None
            classification, regression = calculate_pathway_scores(**args)
        self.assertGreater(len(classification), 0)
        self.assertEqual(len(regression), 0)

    @patch('scripts.calculate_scores.save_csv')
    def test_saves_to_csv_in_batch_mode(self, mock_save_csv):
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '1'}):
            result = calculate_pathway_scores(**self.default_params)
        self.assertIsNone(result)
        self.assertEqual(mock_save_csv.call_count, 2)

    def test_reproducibility_with_seed(self):
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            classification1, regression1 = calculate_pathway_scores(**self.default_params)
            classification2, regression2 = calculate_pathway_scores(**self.default_params)
        pd.testing.assert_frame_equal(classification1, classification2, check_exact=False, rtol=1e-6)
        pd.testing.assert_frame_equal(regression1, regression2, check_exact=False, rtol=1e-6)


class CalculateBackgroundScoresInRealModeTest(Test):

    def setUp(self):
        self.cell_types = ['TypeA', 'TypeB']
        self.lineages = ['Lineage1', 'Lineage2']
        self.sizes = [5, 10, 15]
        self.len_gene_sets = 10
        classification, regression = [], []
        for size in self.sizes:
            for ct in self.cell_types:
                for _ in range(self.len_gene_sets):
                    classification.append({
                        'set_size': size,
                        TARGET_COL: ct,
                        'pathway_score': np.random.uniform(0, 1),
                    })
            for lin in self.lineages:
                for _ in range(self.len_gene_sets):
                    regression.append({
                        'set_size': size,
                        TARGET_COL: lin,
                        'pathway_score': np.random.uniform(0, 1),
                    })
        self.classification = pd.DataFrame(classification)
        self.regression = pd.DataFrame(regression)
    
    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_real_background_scores_happy_flow(self, mock_save_background_scores):
        calculate_background_scores_in_real_mode(
            tmp='',
            cache='',
            classification=self.classification,
            regression=self.regression,
            trim_background=False,
        )
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * (len(self.cell_types) + len(self.lineages)))
        for call in mock_save_background_scores.call_args_list:
            assert len(call.args[0]) == self.len_gene_sets

    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_real_background_scores_without_classification(self, mock_save_background_scores):
        calculate_background_scores_in_real_mode(
            tmp='',
            cache='',
            classification=None,
            regression=self.regression,
        )
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * len(self.lineages))

    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_real_background_scores_without_regression(self, mock_save_background_scores):
        calculate_background_scores_in_real_mode(
            tmp='',
            cache='',
            classification=self.classification,
            regression=None,
        )
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * len(self.cell_types))

    @patch('scripts.output.read_results')
    def test_aggregate_batch_results(self, mock_read_results):
        with tempfile.TemporaryDirectory() as tmp:
            result_type = 'cell_type_classification'
            open(os.path.join(tmp, f"{result_type}_batch1.csv"), "w").close()
            open(os.path.join(tmp, f"{result_type}_batch2.csv"), "w").close()
            open(os.path.join(tmp, f"{result_type}_batch3.csv"), "w").close()

            def side_effect(filename, tmp_dir, index_col=None):
                if filename.endswith("batch1.csv"):
                    return pd.DataFrame({"batch": [1]})
                if filename.endswith("batch2.csv"):
                    return pd.DataFrame({"batch": [2]})
                if filename.endswith("batch3.csv"):
                    return pd.DataFrame({"batch": [3]})
                return None

            mock_read_results.side_effect = side_effect
            result = aggregate_batch_results(tmp=tmp, result_type=result_type)

            self.assertEqual(len(result), 3)
            self.assertCountEqual(result["batch"].tolist(), [1, 2, 3])


class CalculateBackgroundScoresInRandomModeTest(Test):

    def setUp(self):
        self.expression = self.generate_data(num_cells=20, num_genes=30)
        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA'] * 10 + ['TypeB'] * 10,
        }, index=self.expression.index)
        self.pseudotime = pd.DataFrame({
            'Lineage1': np.linspace(0, 1, 20),
            'Lineage2': np.random.uniform(0, 1, 20),
        }, index=self.expression.index)
        self.sizes = [5, 10, 15]

    def test_get_target_size_pair_batch(self):
        sizes = [10, 20, 30]
        targets = ["A", "B", "C"]
        all_pairs = [
            (10, "A"), (10, "B"), (10, "C"),
            (20, "A"), (20, "B"), (20, "C"),
            (30, "A"), (30, "B"), (30, "C"),
        ]

        batch_size = 2

        out0 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=0, batch_size=batch_size)
        out1 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=1, batch_size=batch_size)
        out2 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=2, batch_size=batch_size)
        out3 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=3, batch_size=batch_size)
        out4 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=4, batch_size=batch_size)
        out5 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=5, batch_size=batch_size)
        out6 = _get_target_size_pair_batch(sizes=sizes, targets=targets, batch=6, batch_size=batch_size)
        
        self.assertEqual(out0, all_pairs)
        self.assertEqual(out1, all_pairs[0:2])
        self.assertEqual(out2, all_pairs[2:4])
        self.assertEqual(out3, all_pairs[4:6])
        self.assertEqual(out4, all_pairs[6:8])
        self.assertEqual(out5, all_pairs[8:9])
        self.assertEqual(out6, [])

    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_random_background_scores_happy_flow(self, mock_save_background_scores):
        repeats = 3
        calculate_background_scores_in_random_mode(
            repeats=repeats,
            sizes=self.sizes,
            expression=self.expression,
            cell_types=self.cell_types,
            pseudotime=self.pseudotime,
            classifier=CLASSIFIER,
            regressor=REGRESSOR,
            classification_metric=CLASSIFICATION_METRIC,
            regression_metric=REGRESSION_METRIC,
            cross_validation=2,
            processes=0,
            output='',
            cache='',
            trim_background=False,
        )
        len_targets = len(self.cell_types[CELL_TYPE_COL].unique()) + len(self.pseudotime.columns) + 1  # plus 'All'
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * len_targets)
        for call in mock_save_background_scores.call_args_list:
            assert len(call.args[0]) == repeats

    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_random_background_scores_without_classification(self, mock_save_background_scores):
        calculate_background_scores_in_random_mode(
            repeats=1,
            sizes=self.sizes,
            expression=self.expression,
            cell_types=None,
            pseudotime=self.pseudotime,
            classifier=CLASSIFIER,
            regressor=REGRESSOR,
            classification_metric=CLASSIFICATION_METRIC,
            regression_metric=REGRESSION_METRIC,
            cross_validation=2,
            processes=0,
            output='',
            cache='',
        )
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * len(self.pseudotime.columns))

    @patch('scripts.calculate_scores.save_background_scores')
    def test_calculate_random_background_scores_without_regression(self, mock_save_background_scores):
        calculate_background_scores_in_random_mode(
            repeats=1,
            sizes=self.sizes,
            expression=self.expression,
            cell_types=self.cell_types,
            pseudotime=None,
            classifier=CLASSIFIER,
            regressor=REGRESSOR,
            classification_metric=CLASSIFICATION_METRIC,
            regression_metric=REGRESSION_METRIC,
            cross_validation=2,
            processes=0,
            output='',
            cache='',
            trim_background=False,
        )
        self.assertEqual(mock_save_background_scores.call_count, len(self.sizes) * (len(self.cell_types[CELL_TYPE_COL].unique()) + 1))  # plus 'All'


if __name__ == '__main__':
    unittest.main()
