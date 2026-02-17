import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from scripts.step_4_aggregation import calculate_p_value, evaluate_and_correct_result, aggregate
from scripts.consts import TARGET_COL, BackgroundMode
from tests.interface import Test


class CalculatePValueTest(Test):

    def test_uses_mem_cache_on_second_call(self):
        """Verify mem_cache is used to avoid reloading background scores."""
        mem_cache = {}

        with patch('scripts.step_4_aggregation.load_background_scores', return_value=[0.5, 0.6, 0.7, 0.8]) as mock_load:
            # First call - should load from disk
            calculate_p_value(
                pathway_score=0.9,
                distribution='normal',
                set_size=10,
                background_mode=BackgroundMode.REAL,
                cache='/fake/cache',
                mem_cache=mem_cache,
                cell_type='TypeA',
            )
            self.assertEqual(mock_load.call_count, 1)

            # Second call with same params - should use cache
            calculate_p_value(
                pathway_score=0.85,
                distribution='normal',
                set_size=10,
                background_mode=BackgroundMode.REAL,
                cache='/fake/cache',
                mem_cache=mem_cache,
                cell_type='TypeA',
            )
            self.assertEqual(mock_load.call_count, 1)  # Still 1, not 2

    def test_different_background_keys_load_separately(self):
        """Different (size, target) combinations should load separately."""
        mem_cache = {}

        with patch('scripts.step_4_aggregation.load_background_scores', return_value=[0.5, 0.6, 0.7]):
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=10,
                background_mode=BackgroundMode.REAL, cache='', mem_cache=mem_cache, cell_type='TypeA',
            )
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=10,
                background_mode=BackgroundMode.REAL, cache='', mem_cache=mem_cache, cell_type='TypeB',
            )
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=20,
                background_mode=BackgroundMode.REAL, cache='', mem_cache=mem_cache, cell_type='TypeA',
            )
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=10,
                background_mode=BackgroundMode.RANDOM, cache='', mem_cache=mem_cache,
                cell_type='TypeA', repeats=100,
            )
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=10,
                background_mode=BackgroundMode.RANDOM, cache='', mem_cache=mem_cache,
                cell_type='TypeA', repeats=150,
            )
            # Identical to previous call
            calculate_p_value(
                pathway_score=0.9, distribution='normal', set_size=10,
                background_mode=BackgroundMode.RANDOM, cache='', mem_cache=mem_cache,
                cell_type='TypeA', repeats=150,
            )
            self.assertEqual(len(mem_cache), 5)

    def test_background_score_mean_calculation(self):
        """Verify background score mean is calculated correctly."""
        background_scores = [0.2, 0.4, 0.6, 0.8]
        with patch('scripts.step_4_aggregation.load_background_scores', return_value=background_scores):
            _, bg_mean = calculate_p_value(
                pathway_score=0.5,
                distribution='normal',
                set_size=10,
                background_mode=BackgroundMode.REAL,
                cache='',
                mem_cache={},
                cell_type='TypeA',
            )
            self.assertAlmostEqual(bg_mean, np.mean(background_scores))


class EvaluateAndCorrectResultTest(Test):

    def _create_result_df(self, targets: list[str], scores: list[float], sizes: list[int]) -> pd.DataFrame:
        return pd.DataFrame({
            TARGET_COL: targets,
            'pathway_score': scores,
            'set_size': sizes,
            'set_name': ['PathwayA'] * len(targets),
            'effect_size': [0.3] * len(targets),
        })

    @patch('scripts.step_4_aggregation.save_csv')
    @patch('scripts.step_4_aggregation.load_background_scores', return_value=[0.4, 0.5, 0.6])
    def test_adds_statistical_columns(self, mock_load, mock_save):
        """Verify p_value, fdr, background_scores, background_score_mean, corrected_effect_size are added."""
        result = self._create_result_df(['TypeA', 'TypeB'], [0.8, 0.9], [10, 10])
        output = evaluate_and_correct_result(
            result=result,
            result_type='cell_type_classification',
            background_mode=BackgroundMode.REAL,
            distribution='normal',
            output='', tmp='', cache='', repeats=100,
            corrected_effect_size=True,
        )
        expected_cols = ['p_value', 'fdr', 'background_score_mean', 'corrected_effect_size']
        for col in expected_cols:
            self.assertIn(col, output.columns)

    @patch('scripts.step_4_aggregation.save_csv')
    def test_classification_uses_cell_type_for_background(self, mock_save):
        """cell_type_classification should use TARGET_COL as cell_type param."""
        result = self._create_result_df(['TypeA'], [0.8], [10])
        with patch('scripts.step_4_aggregation.calculate_p_value', wraps=calculate_p_value) as mock_calc:
            mock_calc.return_value = (0.05, 0.55)
            evaluate_and_correct_result(
                result=result,
                result_type='cell_type_classification',
                background_mode=BackgroundMode.REAL,
                distribution='normal',
                output='', tmp='', cache='', repeats=100,
                corrected_effect_size=True,
            )
            call_kwargs = mock_calc.call_args[1]
            self.assertEqual(call_kwargs['cell_type'], 'TypeA')
            self.assertIsNone(call_kwargs['lineage'])

    @patch('scripts.step_4_aggregation.save_csv')
    def test_regression_uses_lineage_for_background(self, mock_save):
        """pseudotime_regression should use TARGET_COL as lineage param."""
        result = self._create_result_df(['Lineage1'], [0.8], [10])
        with patch('scripts.step_4_aggregation.calculate_p_value', wraps=calculate_p_value) as mock_calc:
            mock_calc.return_value = (0.05, 0.55)
            evaluate_and_correct_result(
                result=result,
                result_type='pseudotime_regression',
                background_mode=BackgroundMode.REAL,
                distribution='normal',
                output='', tmp='', cache='', repeats=100,
                corrected_effect_size=True,
            )
            call_kwargs = mock_calc.call_args[1]
            self.assertIsNone(call_kwargs['cell_type'])
            self.assertEqual(call_kwargs['lineage'], 'Lineage1')


class AggregateTest(Test):

    @patch('scripts.step_4_aggregation.load_background_scores', return_value=[0.4, 0.5, 0.6])
    @patch('scripts.step_4_aggregation.save_csv')
    @patch('scripts.step_4_aggregation.plot')
    def test_regression(self, mock_plot, mock_save, mock_load):
        classification = pd.DataFrame({
            TARGET_COL: ['TypeA', 'TypeB'],
            'pathway_score': [0.8, 0.9],
            'set_size': [10, 10],
            'set_name': ['PathwayA', 'PathwayA'],
            'effect_size': [0.3, 0.4],
        })
        regression = pd.DataFrame({
            TARGET_COL: ['Lineage1', 'Lineage2'],
            'pathway_score': [0.7, 0.85],
            'set_size': [15, 15],
            'set_name': ['PathwayB', 'PathwayB'],
            'effect_size': [0.25, 0.35],
        })
        aggregate(
            output='', tmp='', cache='',
            background_mode=BackgroundMode.REAL,
            distribution='normal',
            repeats=100,
            classification=classification,
            regression=regression,
            corrected_effect_size=True,
            verbose=False,
        )


if __name__ == '__main__':
    unittest.main()
