import unittest
import tempfile
import os
import gzip
import pandas as pd
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
from tests.interface import Test
from scripts.data import (
    preprocess_expression,
    preprocess_data,
    reduce_dimension,
    scale_expression,
    scale_pseudotime,
    sum_gene_expression,
    mean_gene_expression,
    calculate_cell_type_effect_size,
    _get_lineage_info,
    calculate_pseudotime_effect_size,
    transform_log,
    re_transform_log
)
from scripts.consts import CELL_TYPE_COL, TARGET_COL
from scripts.output import _read_10x_mtx


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


class GeneExpressionAggregationTest(Test):
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


class EffectSizeTest(Test):

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

        effect_size = calculate_cell_type_effect_size(results, expression, cell_types)

        mean_target1 = np.mean([np.mean([1, 6]), np.mean([2, 7])])
        mean_other1 = np.mean([np.mean([3, 8]), np.mean([4, 9]), np.mean([5, 10])])
        assert effect_size[0] == mean_target1 - mean_other1

        mean_target2 = np.mean([np.mean([13, 18, 23]), np.mean([14, 19, 24])])
        mean_other2 = np.mean([np.mean([11, 16, 21]), np.mean([12, 17, 22]), np.mean([15, 20, 25])])
        assert effect_size[1] == mean_target2 - mean_other2

    def test_get_lineage_info(self):
        # synthetic data
        cells = [f"c{i}" for i in range(10)]
        pseudotime = pd.DataFrame(
            {"L1": np.linspace(0, 1, 10)},
            index=cells
        )

        masked_expression = pd.DataFrame(
            {
                "g0": np.arange(10, dtype=float),
                "g1": np.ones(10, dtype=float),
            },
            index=cells
        )

        info = _get_lineage_info(
            masked_expression=masked_expression,
            targets=["L1"],
            pseudotime=pseudotime,
            percentile=0.2,  # size = 2
            bins=3,
            delta=0.05,
        )["L1"]

        assert set(info.keys()) == {"orig_cells", "bins_cells", "bins_mean_pt"}

        # orig_cells should be mean of first 2 cells
        expected_orig = masked_expression.iloc[:2].mean()
        pd.testing.assert_series_equal(info["orig_cells"], expected_orig)

        # bins lists aligned
        assert len(info["bins_cells"]) == len(info["bins_mean_pt"])
        # last bin (mean close to 1) must be included
        last_mean = pseudotime.iloc[-2:]["L1"].mean()
        assert any(abs(m - last_mean) < 1e-12 for m in info["bins_mean_pt"])

    def test_get_lineage_info_filters_all_intermediate_bins(self):
        # 10 cells, pseudotime 0..1
        cells = [f"c{i}" for i in range(10)]
        pseudotime = pd.DataFrame({"L1": np.linspace(0, 1, 10)}, index=cells)

        masked_expression = pd.DataFrame(
            {"g0": np.arange(10, dtype=float), "g1": np.ones(10, dtype=float)},
            index=cells,
        )

        # With delta larger than any possible distance between intermediate bin means and orig/last,
        # all intermediate bins should be removed by the filtering rules.
        info = _get_lineage_info(
            masked_expression=masked_expression,
            targets=["L1"],
            pseudotime=pseudotime,
            percentile=0.2,  # size=2
            bins=10,
            delta=0.6,       # aggressive: wipes all intermediate means vs orig_mean (~0.0556) and last_mean (~0.9444)
        )["L1"]

        assert info is not None
        assert set(info.keys()) == {"orig_cells", "bins_cells", "bins_mean_pt"}

        # Should end up with only the explicit last bin appended
        assert len(info["bins_cells"]) == 1
        assert len(info["bins_mean_pt"]) == 1

        # And that bin mean should be the last mean (mean of last 20% cells)
        expected_last_mean = pseudotime.iloc[-2:]["L1"].mean()
        assert abs(info["bins_mean_pt"][0] - expected_last_mean) < 1e-12

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

        effect_size, _ = calculate_pseudotime_effect_size(results, expression, pseudotime, percentile=0.3, bins=4)

        # target1: pseudotime values: cell1=0.1, cell2=0.2
        # window size = ceil(2 * 0.3) = 1
        # reference: earliest 30% = cell1 only
        # pt_range: 0.1 to 0.2, bins at [0.125, 0.15, 0.175, 0.2]
        # For all bins, closest cell is cell1 or cell2 (distance 0.025, 0.05, 0.075, 0.1 or 0, 0.025, 0.05, 0.075)
        # All bins select cell2 (closest to all bin points)
        mean_min1 = 6  # cell1, g2
        mean_max1 = 7  # cell2, g2
        assert effect_size[0] == mean_max1 - mean_min1

        # target2: pseudotime values: cell2=0.6, cell4=0.7, cell5=0.8, cell3=0.9
        # window size = ceil(4 * 0.3) = 2
        # reference: earliest 30% = cell2, cell4 (first 2 cells when sorted)
        # pt_range: 0.6 to 0.9, bins at [0.675, 0.75, 0.825, 0.9]
        
        # Reference mean for genes g3, g4, g5 from cell2 and cell4:
        orig_mean = np.mean([np.mean([12, 17, 22]), np.mean([14, 19, 24])])
        
        # bin 0.675: closest 2 cells are cell2 (0.075) and cell4 (0.025)
        mean_bin1 = np.mean([np.mean([12, 17, 22]), np.mean([14, 19, 24])])
        # bin 0.75: closest 2 cells are cell4 (0.05) and cell5 (0.05) 
        mean_bin2 = np.mean([np.mean([14, 19, 24]), np.mean([15, 20, 25])])
        # bin 0.825: closest 2 cells are cell5 (0.025) and cell4 (0.125) or cell3 (0.075)
        mean_bin3 = np.mean([np.mean([15, 20, 25]), np.mean([13, 18, 23])])
        # bin 0.9: closest 2 cells are cell3 (0.0) and cell5 (0.1)
        mean_bin4 = np.mean([np.mean([13, 18, 23]), np.mean([15, 20, 25])])
        
        max_diff = max(abs(mean_bin1 - orig_mean), abs(mean_bin2 - orig_mean), 
                      abs(mean_bin3 - orig_mean), abs(mean_bin4 - orig_mean))
        if abs(mean_bin2 - orig_mean) == max_diff:
            max_mean = mean_bin2
        elif abs(mean_bin3 - orig_mean) == max_diff:
            max_mean = mean_bin3
        elif abs(mean_bin4 - orig_mean) == max_diff:
            max_mean = mean_bin4
        else:
            max_mean = mean_bin1
        
        assert effect_size[1] == max_mean - orig_mean


class MTXLoaderTest(Test):

    def test_read_10x_mtx_folder(self):
        # Create a temporary directory for test data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data: 3 genes x 4 cells
            # Matrix in COO format (row, col, data)
            row = [0, 0, 1, 2, 2]  # gene indices
            col = [0, 2, 1, 0, 3]  # cell indices
            data = [1.5, 2.0, 3.5, 4.0, 5.5]
            
            matrix = coo_matrix((data, (row, col)), shape=(3, 4))
            
            # Write matrix.mtx
            matrix_path = os.path.join(tmpdir, 'matrix.mtx')
            mmwrite(matrix_path, matrix)
            
            # Write features.tsv (3 columns: gene_id, gene_name, feature_type)
            features_path = os.path.join(tmpdir, 'features.tsv')
            with open(features_path, 'w') as f:
                f.write('ENSG001\tGENE1\tGene Expression\n')
                f.write('ENSG002\tGENE2\tGene Expression\n')
                f.write('ENSG003\tGENE3\tGene Expression\n')
            
            # Write barcodes.tsv
            barcodes_path = os.path.join(tmpdir, 'barcodes.tsv')
            with open(barcodes_path, 'w') as f:
                f.write('AAACCTGAGAAACCAT-1\n')
                f.write('AAACCTGAGAAACCGC-1\n')
                f.write('AAACCTGAGAAACCTA-1\n')
                f.write('AAACCTGAGAAACGAG-1\n')
            
            # Load the data
            df = _read_10x_mtx(tmpdir)
            
            # Assertions
            assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
            assert df.shape == (4, 3), f"Expected shape (4, 3), got {df.shape}"
            
            # Check gene names (columns)
            expected_genes = ['GENE1', 'GENE2', 'GENE3']
            assert list(df.columns) == expected_genes, f"Expected columns {expected_genes}, got {list(df.columns)}"
            
            # Check barcode names (rows)
            expected_barcodes = [
                'AAACCTGAGAAACCAT-1',
                'AAACCTGAGAAACCGC-1',
                'AAACCTGAGAAACCTA-1',
                'AAACCTGAGAAACGAG-1'
            ]
            assert list(df.index) == expected_barcodes, f"Expected index {expected_barcodes}, got {list(df.index)}"
            
            # Check specific values (matrix is transposed: cells x genes)
            assert df.loc['AAACCTGAGAAACCAT-1', 'GENE1'] == 1.5
            assert df.loc['AAACCTGAGAAACCTA-1', 'GENE1'] == 2.0
            assert df.loc['AAACCTGAGAAACCGC-1', 'GENE2'] == 3.5
            assert df.loc['AAACCTGAGAAACCAT-1', 'GENE3'] == 4.0
            assert df.loc['AAACCTGAGAAACGAG-1', 'GENE3'] == 5.5
            
            # Check zeros
            assert df.loc['AAACCTGAGAAACCGC-1', 'GENE1'] == 0.0
            assert df.loc['AAACCTGAGAAACCAT-1', 'GENE2'] == 0.0

    def test_read_10x_mtx_folder_gzipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data: 2 genes x 3 cells
            row = [0, 1, 1]
            col = [0, 1, 2]
            data = [1.0, 2.0, 3.0]
            
            matrix = coo_matrix((data, (row, col)), shape=(2, 3))
            
            # Write gzipped matrix.mtx.gz
            matrix_path = os.path.join(tmpdir, 'matrix.mtx')
            mmwrite(matrix_path, matrix)
            with open(matrix_path, 'rb') as f_in:
                with gzip.open(matrix_path + '.gz', 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(matrix_path)  # Remove uncompressed version
            
            # Write gzipped features.tsv.gz
            features_path = os.path.join(tmpdir, 'features.tsv.gz')
            with gzip.open(features_path, 'wt') as f:
                f.write('ENSG001\tGENEA\tGene Expression\n')
                f.write('ENSG002\tGENEB\tGene Expression\n')
            
            # Write gzipped barcodes.tsv.gz
            barcodes_path = os.path.join(tmpdir, 'barcodes.tsv.gz')
            with gzip.open(barcodes_path, 'wt') as f:
                f.write('BARCODE1-1\n')
                f.write('BARCODE2-1\n')
                f.write('BARCODE3-1\n')
            
            # Load the data
            df = _read_10x_mtx(tmpdir)
            
            # Assertions
            assert df.shape == (3, 2), f"Expected shape (3, 2), got {df.shape}"
            assert list(df.columns) == ['GENEA', 'GENEB']
            assert df.loc['BARCODE1-1', 'GENEA'] == 1.0
            assert df.loc['BARCODE2-1', 'GENEB'] == 2.0
            assert df.loc['BARCODE3-1', 'GENEB'] == 3.0

    def test_read_10x_mtx_folder_feature_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create matrix: 2 genes x 3 cells
            row = [0, 1]
            col = [0, 1]
            data = [1.0, 2.0]
            matrix = coo_matrix((data, (row, col)), shape=(2, 3))
            
            matrix_path = os.path.join(tmpdir, 'matrix.mtx')
            mmwrite(matrix_path, matrix)
            
            # Write features.tsv with WRONG number of genes (3 instead of 2)
            features_path = os.path.join(tmpdir, 'features.tsv')
            with open(features_path, 'w') as f:
                f.write('ENSG001\tGENEA\tGene Expression\n')
                f.write('ENSG002\tGENEB\tGene Expression\n')
                f.write('ENSG003\tGENEC\tGene Expression\n')  # Extra gene
            
            # Write barcodes.tsv
            barcodes_path = os.path.join(tmpdir, 'barcodes.tsv')
            with open(barcodes_path, 'w') as f:
                f.write('BARCODE1-1\n')
                f.write('BARCODE2-1\n')
                f.write('BARCODE3-1\n')
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                _read_10x_mtx(tmpdir)
            
            self.assertIn("Matrix rows", str(context.exception))
            self.assertIn("number of genes", str(context.exception))

    def test_read_10x_mtx_folder_barcode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create matrix: 2 genes x 3 cells
            row = [0, 1]
            col = [0, 1]
            data = [1.0, 2.0]
            matrix = coo_matrix((data, (row, col)), shape=(2, 3))
            
            matrix_path = os.path.join(tmpdir, 'matrix.mtx')
            mmwrite(matrix_path, matrix)
            
            # Write features.tsv
            features_path = os.path.join(tmpdir, 'features.tsv')
            with open(features_path, 'w') as f:
                f.write('ENSG001\tGENEA\tGene Expression\n')
                f.write('ENSG002\tGENEB\tGene Expression\n')
            
            # Write barcodes.tsv with WRONG number of barcodes (2 instead of 3)
            barcodes_path = os.path.join(tmpdir, 'barcodes.tsv')
            with open(barcodes_path, 'w') as f:
                f.write('BARCODE1-1\n')
                f.write('BARCODE2-1\n')
                # Missing BARCODE3-1
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                _read_10x_mtx(tmpdir)
            
            self.assertIn("Matrix columns", str(context.exception))
            self.assertIn("number of barcodes", str(context.exception))


if __name__ == '__main__':
    unittest.main()
