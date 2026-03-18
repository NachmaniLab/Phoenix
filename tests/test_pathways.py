import unittest
import pandas as pd
from tests.interface import Test
from scripts.pathways import get_kegg_organism, retrieve_all_kegg_pathways, retrieve_all_go_pathways, retrieve_all_msigdb_pathways, intersect_genes
from scripts.visualization import get_top_sum_pathways, get_column_unique_pathways, filter_top_pathways
from scripts.consts import (
    TARGET_COL, ALL_CELLS,
    FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD,
    IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD,
)

class KeggTest(Test):
    
    def test_organism_name(self):
        assert get_kegg_organism('homo sapiens') == 'hsa'

    def test_pathway_retrieval(self):
        pathways = retrieve_all_kegg_pathways('human')  # uses MSigDB
        assert len(pathways) > 500

        pathways = retrieve_all_kegg_pathways('empedobacter brevis', subset=15)
        assert len(pathways) > 2


class GoTest(Test):
    
    def test_pathway_retrieval(self):
        try:
            pathways = retrieve_all_go_pathways('human')
            assert len(pathways) > 5000
        except RuntimeError as e:
            if 'Error getting the Enrichr libraries' in str(e):
                pass
            else:
                raise


class MsigdbTest(Test):
    
    def test_pathway_retrieval(self):
        pathways = retrieve_all_msigdb_pathways('human')
        assert len(pathways) > 30000


class GeneDataTest(Test):

    def setUp(self):
        self.all_genes = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6', 'Gene7', 'Gene8']

    def test_gene_intersection(self):
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene8'], self.all_genes), ['Gene1', 'Gene4', 'Gene8'])
        self.assertEqual(intersect_genes(['GENE1', 'GENE4', 'GENE8'], self.all_genes), ['Gene1', 'Gene4', 'Gene8'])
        self.assertEqual(intersect_genes(['Gene9', 'Gene10'], self.all_genes), [])
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene10'], self.all_genes), ['Gene1', 'Gene4'])
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene4'], self.all_genes), ['Gene1', 'Gene4'])

    def test_top_sum_pathways(self):
        df = pd.DataFrame({
            'Target1': [0.1, 0.2, 0.3],
            'Target2': [0.4, None, 0.6],
            'Target3': [0.7, 0.8, 0.9]
        }, index=['Pathway1', 'Pathway2', 'Pathway3'])

        result = get_top_sum_pathways(df, ascending=False, size=2)
        expected = ['Pathway3', 'Pathway1']
        self.assertEqual(result, expected)

        result = get_top_sum_pathways(df, ascending=True, size=2)
        expected = ['Pathway1', 'Pathway3']
        self.assertEqual(result, expected)

    def test_column_unique_pathways(self):
        """
        Pathway5 is above threshold, Pathway6 is not the minimum
        """
        df = pd.DataFrame({
            'Target1': [0.1, 0.2, 0.3, 0.4, 0.6, 0.2],
            'Target2': [0.4, None, 0.6, 0.7, 0.8, 0.1],
            'Target3': [0.7, 0.8, 0.9, 0.5, 0.6, 0.2],
            'Target4': [0.2, 0.3, 0.6, 0.6, 0.7, 0.2]
        }, index=['Pathway1', 'Pathway2', 'Pathway3', 'Pathway4', 'Pathway5', 'Pathway6'])

        result = get_column_unique_pathways(df, 'Target1', size=10, threshold=0.5)
        expected = ['Pathway3', 'Pathway1', 'Pathway2', 'Pathway4']
        self.assertEqual(result, expected)


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
        results = pd.DataFrame([
            self._make_passing_row('TypeA', 'Pathway1'),
            self._make_passing_row('TypeB', 'Pathway2'),
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 2)
        self.assertListEqual(list(out.columns), [TARGET_COL, 'set_name'])

    def test_fdr_filter(self):
        """Rows with fdr above the threshold should be excluded."""
        results = pd.DataFrame([
            self._make_passing_row('TypeA', 'Pathway1'),
            {**self._make_passing_row('TypeA', 'Pathway2'), 'fdr': 0.1},  # fails FDR
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]['set_name'], 'Pathway1')

    def test_corrected_effect_size_filter_positive(self):
        """Rows with |corrected_effect_size| below threshold should be excluded."""
        results = pd.DataFrame([
            self._make_passing_row('TypeA', 'Pathway1'),
            {**self._make_passing_row('TypeA', 'Pathway2'), 'corrected_effect_size': 0.5},  # fails |es|
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]['set_name'], 'Pathway1')

    def test_corrected_effect_size_filter_negative(self):
        """Negative corrected_effect_size with sufficient magnitude should pass."""
        results = pd.DataFrame([
            {**self._make_passing_row('TypeA', 'Pathway1'), 'corrected_effect_size': -1.5},
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]['set_name'], 'Pathway1')

    def test_corrected_effect_size_filter_negative_small(self):
        """Negative corrected_effect_size below threshold (by magnitude) should be excluded."""
        results = pd.DataFrame([
            {**self._make_passing_row('TypeA', 'Pathway1'), 'corrected_effect_size': -0.5},
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 0)

    def test_importance_filter_majority_below_threshold(self):
        """Pathway where > 50% of genes have importance < 0.05 should be excluded."""
        # 3 out of 4 genes have importance < 0.05 → 75% → excluded
        results = pd.DataFrame([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.01; 0.02; 0.01; 0.4'},
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 0)

    def test_importance_filter_exactly_half(self):
        """Pathway where exactly 50% of genes have importance < 0.05 should be kept."""
        # 1 out of 2 genes has importance < 0.05 → 50% → kept (fraction <= threshold)
        results = pd.DataFrame([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.01; 0.4'},
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)

    def test_importance_filter_all_above(self):
        """Pathway where all genes have importance >= 0.05 should be kept."""
        results = pd.DataFrame([
            {**self._make_passing_row('TypeA', 'Pathway1'),
             'gene_importances': '0.1; 0.2; 0.3'},
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)

    def test_all_cells_excluded(self):
        """Rows with target == ALL_CELLS should always be excluded."""
        results = pd.DataFrame([
            {**self._make_passing_row(ALL_CELLS, 'Pathway1')},
            self._make_passing_row('TypeA', 'Pathway2'),
        ])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0][TARGET_COL], 'TypeA')

    def test_empty_results(self):
        """Empty input should produce empty output."""
        results = pd.DataFrame(columns=[TARGET_COL, 'set_name', 'fdr', 'corrected_effect_size', 'gene_importances'])
        out = filter_top_pathways(results, FDR_THRESHOLD, CORRECTED_EFFECT_SIZE_THRESHOLD, IMPORTANCE_LOWER_THRESHOLD, IMPORTANCE_GENE_FRACTION_THRESHOLD, 'corrected_effect_size')
        self.assertEqual(len(out), 0)
        self.assertListEqual(list(out.columns), [TARGET_COL, 'set_name'])


if __name__ == '__main__':
    unittest.main()
