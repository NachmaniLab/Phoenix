import unittest
import pandas as pd
from tests.interface import Test
from scripts.pathways import get_kegg_organism, retrieve_all_kegg_pathways, retrieve_all_go_pathways, retrieve_all_msigdb_pathways, intersect_genes
from scripts.visualization import get_top_sum_pathways, get_column_unique_pathways


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


if __name__ == '__main__':
    unittest.main()
