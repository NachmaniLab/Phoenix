import unittest
import pandas as pd
import numpy as np
from tests.interface import Test
from scripts.data import preprocess_expression, preprocess_data, reduce_dimension, scale_expression, scale_pseudotime, sum_gene_expression, mean_gene_expression, calculate_cell_type_effect_size, calculate_pseudotime_effect_size, transform_log, re_transform_log
from scripts.consts import CELL_TYPE_COL, TARGET_COL


class PreprocessingTest(Test):

    def setUp(self) -> None:
        self.expression = self.generate_data(3, 3)

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['Type1', 'Type2', 'Type1']
        }, index=self.expression.index)

        self.pseudotime = pd.DataFrame({
            'Lineage1': [0.1, 0.2, np.nan],
            'Lineage2': [0.4, 0.5, 0.6]
        }, index=self.expression.index)

        self.reduction = pd.DataFrame({
            'UMAP1': [0.1, 0.2, 0.3],
            'UMAP2': [0.4, 0.5, 0.6]
        }, index=self.expression.index)

    def test_preprocessing_flow(self):
        preprocess_data(self.expression, self.cell_types, self.pseudotime, self.reduction, verbose=False)

    def test_reduction_flow(self):
        num_cells = 31
        expression = self.generate_data(7, num_cells)
        reduction = reduce_dimension(expression, reduction_method='umap', seed=42, verbose=False)
        assert reduction.shape == (num_cells, 2)
        reduction = reduce_dimension(expression, reduction_method='tsne', seed=42, verbose=False)
        assert reduction.shape == (num_cells, 2)

    def test_gene_filtering(self):
        expression = self.generate_data(num_genes=10, mean=5, std=1)
        lowly_expressed = ['Gene3', 'Gene6']
        expression[lowly_expressed] = 1
        expression = preprocess_expression(expression, preprocessed=True, num_genes=8, verbose=False)
        for gene in lowly_expressed:
            assert gene not in expression.columns

    def test_filter_cell_types_and_pseudotime(self):
        expression = pd.DataFrame({
            "gene1": [1, 2, 3],
            "gene2": [4, 5, 6]
        }, index=["cell1", "cell2", "cell3"])
        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ["T", "B", "T"]
        }, index=["cell1", "cell2", "cell3"])
        pseudotime = pd.DataFrame({
            "lin1": [0.1, 0.2, 0.3],
            "lin2": [0.4, 0.5, 0.6]
        }, index=["cell1", "cell2", "cell3"])
        reduction = pd.DataFrame({
            "pca1": [0.1, 0.2, 0.3],
            "pca2": [0.4, 0.5, 0.6]
        }, index=["cell1", "cell2", "cell3"])

        expr, ct, pt, red = preprocess_data(
            expression=expression,
            cell_types=cell_types,
            pseudotime=pseudotime,
            reduction=reduction,
            exclude_cell_types=["B"],
            exclude_lineages=["lin2"],
            preprocessed=True,
            verbose=False
        )

        expected_index = ["cell1", "cell3"]
        assert list(expr.index) == expected_index
        assert list(ct.index) == expected_index
        assert list(pt.index) == expected_index
        assert list(red.index) == expected_index

        assert "lin2" not in pt.columns
        assert "lin1" in pt.columns

    def test_scaled_expression(self):
        expression = pd.DataFrame({
            'Gene1': [1, 2, 3],
            'Gene2': [4, 5, 6],
            'Gene3': [7, 8, 9]
        }, index=['Cell1', 'Cell2', 'Cell3'])
        scaled_expression = scale_expression(expression)
        assert all(scaled_expression.index == expression.index) and all(scaled_expression.columns == expression.columns)
        assert all(scaled_expression.mean() == 0) and all(round(scaled_expression.std(ddof=0), 7) == 1)
        assert scaled_expression.iloc[1, 1] == (5 - np.mean([4, 5, 6])) / np.std([4, 5, 6])
        assert scaled_expression.iloc[1, 2] == (8 - np.mean([7, 8, 9])) / np.std([7, 8, 9])

    def test_scaled_pseudotime(self):
        pseudotime = pd.DataFrame({
            'Lineage1': [0.1, 0.2, np.nan],
            'Lineage2': [0.4, 0.5, 0.6]
        }, index=['Cell1', 'Cell2', 'Cell3'])
        scaled_pseudotime = scale_pseudotime(pseudotime)
        assert all(scaled_pseudotime.index == pseudotime.index) and all(scaled_pseudotime.columns == pseudotime.columns)
        assert all(scaled_pseudotime.min() == 0) and all(scaled_pseudotime.max() == 1)
        assert scaled_pseudotime.iloc[1, 1] == (0.5 - np.min([0.4, 0.5, 0.6])) / (np.max([0.4, 0.5, 0.6]) - np.min([0.4, 0.5, 0.6]))
        assert scaled_pseudotime.iloc[1, 0] == (0.2 - np.min([0.1, 0.2])) / (np.max([0.1, 0.2]) - np.min([0.1, 0.2]))

    def test_sum_gene_expression_geometric(self):
        df = pd.DataFrame({
            "gene1": [1, 2],
            "gene2": [3, 4]
        }, index=["cell1", "cell2"])
        result = sum_gene_expression(df, geometric=True)
        expected = pd.Series([4, 6], index=["cell1", "cell2"])
        pd.testing.assert_series_equal(result, expected)

        s = pd.Series([1, 2, 3])
        result = sum_gene_expression(s, geometric=True)
        expected = pd.Series([6]).squeeze()
        assert result == expected

    def test_sum_gene_expression_log_transformed(self):
        df = pd.DataFrame({
            "gene1": [1, 2],
            "gene2": [3, 4]
        }, index=["cell1", "cell2"])
        result = sum_gene_expression(df, geometric=False)
        expected_values = pd.Series([
            transform_log(re_transform_log(1) + re_transform_log(3)),
            transform_log(re_transform_log(2) + re_transform_log(4))
        ], index=["cell1", "cell2"])
        pd.testing.assert_series_equal(result, expected_values)

    def test_sum_gene_expression_all_zeros(self):
        df = pd.DataFrame({
            "gene1": [0.0, 0.0],
            "gene2": [0.0, 0.0]
        }, index=["cell1", "cell2"])
        result = sum_gene_expression(df, geometric=False)
        expected = pd.Series([0.0, 0.0], index=["cell1", "cell2"])
        pd.testing.assert_series_equal(result, expected)

    def test_mean_gene_expression_without_filtering(self):
        df = pd.DataFrame({
            "gene1": [1.0, 2.0],
            "gene2": [3.0, 4.0]
        }, index=["cell1", "cell2"])
        
        result = mean_gene_expression(df)
        expected = pd.Series([2.0, 3.0], index=["cell1", "cell2"])
        pd.testing.assert_series_equal(result, expected)

    def test_mean_gene_expression_with_df(self):
        df = pd.DataFrame({
            "gene1": [0.0, 2.0],
            "gene2": [4.0, 0.005]
        }, index=["cell1", "cell2"])
        df = df.mask(df < 0.01)  

        result = mean_gene_expression(df)
        expected = pd.Series([4.0, 2.0], index=["cell1", "cell2"])  # only one non-masked value per row
        pd.testing.assert_series_equal(result, expected)

    def test_mean_gene_expression_with_series(self):
        s = pd.Series([5.0, 10.0])
        result = mean_gene_expression(s)
        expected = (5.0 + 10.0) / 2
        assert result == expected

    def test_cell_type_effect_size(self):
        results = pd.DataFrame({
            TARGET_COL: ['target1', 'target2'],
            'top_genes': ['g1; g2', 'g3; g4; g5']
        })
        expression = pd.DataFrame({
            'g1': [1, 2, 3, 4, 5],
            'g2': [6, 7, 8, 9, 10],
            'g3': [11, 12, 13, 14, 15],
            'g4': [16, 17, 18, 19, 20],
            'g5': [21, 22, 23, 24, 25],
            'g6': [26, 27, 28, 29, 30]
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['target1', 'target1', 'target2', 'target2', 'target3']
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])

        effect_size = results.apply(calculate_cell_type_effect_size, axis=1, masked_expression=expression, cell_types=cell_types)

        mean_target1 = np.mean([np.mean([1, 6]), np.mean([2, 7])])
        mean_other1 = np.mean([np.mean([3, 8]), np.mean([4, 9]), np.mean([5, 10])])
        assert effect_size[0] == mean_target1 - mean_other1

        mean_target2 = np.mean([np.mean([13, 18, 23]), np.mean([14, 19, 24])])
        mean_other2 = np.mean([np.mean([11, 16, 21]), np.mean([12, 17, 22]), np.mean([15, 20, 25])])
        assert effect_size[1] == mean_target2 - mean_other2

    def test_pseudotime_effect_size(self):
        results = pd.DataFrame({
            TARGET_COL: ['target1', 'target2'],
            'top_genes': ['g2', 'g3; g4; g5']
        })
        expression = pd.DataFrame({
            'g1': [1, 2, 3, 4, 5],
            'g2': [6, 7, 8, 9, 10],
            'g3': [11, 12, 13, 14, 15],
            'g4': [16, 17, 18, 19, 20],
            'g5': [21, 22, 23, 24, 25],
            'g6': [26, 27, 28, 29, 30]
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        pseudotime = pd.DataFrame({
            'target1': [0.1, 0.2, np.nan, np.nan, np.nan],
            'target2': [np.nan, 0.6, 0.9, 0.7, 0.8],
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])

        effect_size = results.apply(
            calculate_pseudotime_effect_size,
            axis=1,
            masked_expression=expression,
            pseudotime=pseudotime,
            percentile=0.3,
            bins=4  # no bins
        )

        mean_min1 = 6
        mean_max1 = 7
        assert effect_size[0] == mean_max1 - mean_min1

        # window size = 2
        # target2 ordered cells: cell2 -> cell4 -> cell5 -> cell3

        # first window: cell2, cell4
        mean1 = np.mean([np.mean([12, 17, 22]), np.mean([14, 19, 24])])
        # second window: cell4, cell5
        mean2 = np.mean([np.mean([14, 19, 24]), np.mean([15, 20, 25])])
        # third window: cell5, cell3
        mean3 = np.mean([np.mean([15, 20, 25]), np.mean([13, 18, 23])])
        
        orig_mean = mean1
        diff2 = abs(mean2 - orig_mean)
        diff3 = abs(mean3 - orig_mean)
        max_mean = mean2 if diff2 >= diff3 else mean3
        assert effect_size[1] == max_mean - orig_mean


if __name__ == '__main__':
    unittest.main()
