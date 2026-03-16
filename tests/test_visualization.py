import unittest
import pandas as pd
import numpy as np
from tests.interface import Test
from scripts.visualization import filter_top_pathways
from scripts.consts import (
    TARGET_COL, ALL_CELLS,
    FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD,
    IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD,
)


def make_results(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal results DataFrame for testing filter_top_pathways."""
    return pd.DataFrame(rows)


def call_filter(
        results,
        fdr_threshold=FDR_THRESHOLD,
        corrected_effect_size_threshold=CORRECTED_EFFECT_SIZE_THRESHOLD,
        importance_lower_threshold=IMPORTANCE_LOWER_THRESHOLD,
        importance_gene_fraction_threshold=IMPORTANCE_GENE_FRACTION_THRESHOLD,
    ):
    """Helper that always passes all required thresholds explicitly."""
    return filter_top_pathways(results, fdr_threshold, corrected_effect_size_threshold,
                               importance_lower_threshold, importance_gene_fraction_threshold)


class FilterTopPathwaysTest(Test):

    def _make_passing_row(self, target: str = 'TypeA', pathway: str = 'Pathway1') -> dict:
        """Create a row that passes all filters with default thresholds."""
        return {
            TARGET_COL: target,
            'set_name': pathway,
            'fdr': 0.01,                            # passes FDR <= 0.05
            'corrected_effect_size': 1.5,           # passes |es| >= 1.2
            'gene_importances': '0.4; 0.3; 0.3',    # all importances >= 0.05 → 0% below → passes
        }

    def test_all_pass(self):
        """Rows that satisfy all three thresholds should all be returned."""
        results = make_results([
            self._make_passing_row('TypeA', 'Pathway1'),
            self._make_passing_row('TypeB', 'Pathway2'),
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 2)
        self.assertListEqual(list(out.columns), [TARGET_COL, 'set_name'])

    def test_fdr_filter(self):
        """Rows with fdr above the threshold should be excluded."""
        results = make_results([
            self._make_passing_row('TypeA', 'Pathway1'),
            {**self._make_passing_row('TypeA', 'Pathway2'), 'fdr': 0.1},  # fails FDR
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]['set_name'], 'Pathway1')

    def test_corrected_effect_size_filter_positive(self):
        """Rows with |corrected_effect_size| below threshold should be excluded."""
        results = make_results([
            self._make_passing_row('TypeA', 'Pathway1'),
            {**self._make_passing_row('TypeA', 'Pathway2'), 'corrected_effect_size': 0.5},  # fails |es|
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)

    def test_corrected_effect_size_filter_negative(self):
        """Negative corrected_effect_size with sufficient magnitude should pass."""
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'), 'corrected_effect_size': -1.5},
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)

    def test_corrected_effect_size_filter_negative_small(self):
        """Negative corrected_effect_size below threshold (by magnitude) should be excluded."""
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'), 'corrected_effect_size': -0.5},
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 0)

    def test_importance_filter_majority_below_threshold(self):
        """Pathway where > 50% of genes have importance < 0.05 should be excluded."""
        # 3 out of 4 genes have importance < 0.05 → 75% → excluded
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.01; 0.02; 0.01; 0.4'},
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 0)

    def test_importance_filter_exactly_half(self):
        """Pathway where exactly 50% of genes have importance < 0.05 should be kept."""
        # 1 out of 2 genes has importance < 0.05 → 50% → kept (fraction <= threshold)
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.01; 0.4'},
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)

    def test_importance_filter_all_above(self):
        """Pathway where all genes have importance >= 0.05 should be kept."""
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.1; 0.2; 0.3'},
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)

    def test_all_cells_excluded(self):
        """Rows with target == ALL_CELLS should always be excluded."""
        results = make_results([
            {**self._make_passing_row(ALL_CELLS, 'Pathway1')},
            self._make_passing_row('TypeA', 'Pathway2'),
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0][TARGET_COL], 'TypeA')

    def test_empty_results(self):
        """Empty input should produce empty output."""
        results = pd.DataFrame(columns=[TARGET_COL, 'set_name', 'fdr', 'corrected_effect_size', 'gene_importances'])
        out = call_filter(results)
        self.assertEqual(len(out), 0)
        self.assertListEqual(list(out.columns), [TARGET_COL, 'set_name'])

    def test_custom_thresholds(self):
        """Custom threshold values should be respected."""
        results = make_results([
            {**self._make_passing_row('TypeA', 'Pathway1'), 'fdr': 0.08},  # passes fdr_threshold=0.1
        ])
        out = filter_top_pathways(results, 0.1, CORRECTED_EFFECT_SIZE_THRESHOLD,
                                  IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD)
        self.assertEqual(len(out), 1)

        out_strict = filter_top_pathways(results, 0.05, CORRECTED_EFFECT_SIZE_THRESHOLD,
                                         IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD)
        self.assertEqual(len(out_strict), 0)

    def test_output_columns(self):
        """Output DataFrame should contain exactly [TARGET_COL, 'set_name'] columns."""
        results = make_results([self._make_passing_row()])
        out = call_filter(results)
        self.assertListEqual(list(out.columns), [TARGET_COL, 'set_name'])

    def test_multiple_targets_multiple_pathways(self):
        """Filter should work correctly across multiple targets and pathways."""
        results = make_results([
            self._make_passing_row('TypeA', 'Pathway1'),
            {**self._make_passing_row('TypeA', 'Pathway2'), 'fdr': 0.2},   # fails FDR
            self._make_passing_row('TypeB', 'Pathway1'),
            {**self._make_passing_row('TypeB', 'Pathway3'), 'corrected_effect_size': 0.1},  # fails ES
        ])
        out = call_filter(results)
        self.assertEqual(len(out), 2)
        targets = set(out[TARGET_COL].tolist())
        self.assertEqual(targets, {'TypeA', 'TypeB'})


if __name__ == '__main__':
    unittest.main()
