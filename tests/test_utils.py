import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from scripts.consts import ALL_CELLS, SIZES, BackgroundMode
from tests.interface import Test
from scripts.utils import define_set_size, define_batch_size, balance_gene_sets, convert_to_str, convert_from_str, enum2str, str2enum, remove_outliers, correct_effect_size, save_step_runtime, load_total_runtime, format_runtime


class UtilTest(Test):

    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
        self.define_set_size = lambda set_len, set_fraction, min_set_size: define_set_size(set_len, set_fraction, min_set_size, all_sizes=SIZES)
    
    def test_set_size_definition(self):
        self.assertEqual(self.define_set_size(80, 0.25, 10), 20)
        self.assertEqual(self.define_set_size(100, 0.5, 1), 40)  # 50 would be exact, but 40 is the closest
        self.assertEqual(self.define_set_size(20, 0.5, 15), 15)  # min_set_size makes it 15 instead of 10
        self.assertEqual(self.define_set_size(20, 0.5, 16), 15)  # min_set_size makes it 16 instead of 10, but 15 is the closest
        self.assertEqual(self.define_set_size(20, 0.5, 40), 20)  # min_set_size turns 10 into 40 but 40 is bigger than set_len so set_len is selected
        self.assertEqual(self.define_set_size(8, 0.5, 10), 5)  # min_set_size turns 4 into 10 but 10 is bigger than set_len so set_len is selected, but since 8 is not in SIZES, 5 is selected
        self.assertEqual(self.define_set_size(500, 0.5, 10), 200)  # as 250 is not in SIZES

    def test_num_batches_definition(self):
        self.assertEqual(define_batch_size(9, 3), 3)
        self.assertEqual(define_batch_size(10, 3), 4)
        self.assertEqual(define_batch_size(1, 3), 1)
        
    def test_str_conversion(self):
        self.assertEqual(convert_to_str(2.2), '2.2')
        self.assertEqual(convert_to_str([1, 2]), '1; 2')
        self.assertEqual(convert_to_str({'1': 11, '2': [22, 222]}), '1: 11; 2: 22; 222')

        self.assertEqual(convert_from_str('2.2'), 2.2)
        self.assertEqual(convert_from_str('1; 2'), [1, 2])

    def test_enum_conversion(self):
        self.assertEqual(enum2str(BackgroundMode.RANDOM), 'RANDOM')
        self.assertEqual(str2enum(BackgroundMode, 'random'), BackgroundMode.RANDOM)
        self.assertEqual(str2enum(BackgroundMode, 'AUTO'), BackgroundMode.AUTO)

        original_str = 'AUTO'
        self.assertEqual(enum2str(str2enum(BackgroundMode, original_str)), original_str)
        original_enum = BackgroundMode.AUTO
        self.assertEqual(str2enum(BackgroundMode, enum2str(original_enum)), original_enum)

    def test_remove_outliers(self):
        data = [11, 12, 12, 13, 12, 100]  # 100 is an outlier
        result = remove_outliers(data)
        assert all(x in result for x in data if x != 100)

        data = [10, 12, 14, 16, 18]
        assert remove_outliers(data) == data

    def test_correct_effect_size_basic(self):
        effect_sizes = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        targets = pd.Series(['A', 'A', 'A', 'B', 'B', ALL_CELLS])
        corrected = correct_effect_size(effect_sizes, targets)

        expected_A = [-1.0, 0.0, 1.0]  # Mean of A group is 2.0
        expected_B = [-5.0, 5.0]      # Mean of B group is 15.0

        np.testing.assert_allclose(corrected[:3], expected_A, rtol=1e-5)
        np.testing.assert_allclose(corrected[3:5], expected_B, rtol=1e-5)
        self.assertEqual(corrected[5], 30.0)  # ALL_CELLS should remain unchanged

    def test_format_runtime(self):
        self.assertEqual(format_runtime(0), '0h 0m 0s')
        self.assertEqual(format_runtime(59), '0h 0m 59s')
        self.assertEqual(format_runtime(60), '0h 1m 0s')
        self.assertEqual(format_runtime(3661), '1h 1m 1s')
        self.assertEqual(format_runtime(7200), '2h 0m 0s')

    def test_save_and_load_step_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_step_runtime(tmp, 'step1', 10.5)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'runtime_step1.txt')))
            self.assertAlmostEqual(load_total_runtime(tmp, 'step1'), 10.5)

    def test_save_step_runtime_with_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_step_runtime(tmp, 'step2', 5.0, batch=1)
            save_step_runtime(tmp, 'step2', 7.0, batch=2)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'runtime_step2_batch1.txt')))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'runtime_step2_batch2.txt')))
            self.assertAlmostEqual(load_total_runtime(tmp, 'step2'), 12.0)

    def test_load_total_runtime_all_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_step_runtime(tmp, 'step1', 3.0)
            save_step_runtime(tmp, 'step2', 5.0, batch=1)
            save_step_runtime(tmp, 'step2', 4.0, batch=2)
            save_step_runtime(tmp, 'step3', 8.0)
            save_step_runtime(tmp, 'step4', 2.0)
            self.assertAlmostEqual(load_total_runtime(tmp), 22.0)

    def test_load_total_runtime_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_total_runtime(tmp), 0.0)

    def test_save_step_runtime_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_step_runtime(tmp, 'step1', 10.0)
            save_step_runtime(tmp, 'step1', 20.0)
            self.assertAlmostEqual(load_total_runtime(tmp, 'step1'), 20.0)

    def test_balance_gene_sets_no_processes(self):
        """When processes=0 (single batch), gene sets should be returned unchanged."""
        gene_sets = {'a': ['g1', 'g2', 'g3'], 'b': ['g4'], 'c': ['g5', 'g6']}
        result = balance_gene_sets(gene_sets, 0)
        self.assertEqual(list(result.keys()), ['a', 'b', 'c'])

    def test_balance_gene_sets_single_process(self):
        """When processes=1, gene sets should be returned unchanged."""
        gene_sets = {'a': ['g1', 'g2', 'g3'], 'b': ['g4'], 'c': ['g5', 'g6']}
        result = balance_gene_sets(gene_sets, 1)
        self.assertEqual(list(result.keys()), ['a', 'b', 'c'])

    def test_balance_gene_sets_round_robin(self):
        """Round-robin dealing should spread large and small sets across batches."""
        # sizes: big=10, med1=6, med2=5, small1=2, small2=1, tiny=1
        gene_sets = {
            'big':    [f'g{i}' for i in range(10)],
            'med1':   [f'g{i}' for i in range(6)],
            'med2':   [f'g{i}' for i in range(5)],
            'small1': [f'g{i}' for i in range(2)],
            'small2': [f'g{i}' for i in range(1)],
            'tiny':   [f'g{i}' for i in range(1)],
        }
        result = balance_gene_sets(gene_sets, 3)
        keys = list(result.keys())
        batch_size = define_batch_size(len(keys), 3)  # = 2

        batch1_keys = keys[0:batch_size]
        batch2_keys = keys[batch_size:2*batch_size]
        batch3_keys = keys[2*batch_size:]

        batch1_total = sum(len(result[k]) for k in batch1_keys)
        batch2_total = sum(len(result[k]) for k in batch2_keys)
        batch3_total = sum(len(result[k]) for k in batch3_keys)

        # All gene sets should still be present
        self.assertEqual(set(result.keys()), set(gene_sets.keys()))
        for k in gene_sets:
            self.assertEqual(result[k], gene_sets[k])

        # Batches should be roughly balanced (no batch has more than 2x another)
        totals = [batch1_total, batch2_total, batch3_total]
        self.assertLessEqual(max(totals), 2 * min(totals), f'Batches are not balanced: {totals}')

    def test_balance_gene_sets_preserves_all(self):
        """All gene sets and their contents should be preserved after balancing."""
        gene_sets = {f'set{i}': [f'gene{j}' for j in range(i + 1)] for i in range(20)}
        result = balance_gene_sets(gene_sets, 4)
        self.assertEqual(set(result.keys()), set(gene_sets.keys()))
        for k in gene_sets:
            self.assertEqual(result[k], gene_sets[k])


if __name__ == '__main__':
    unittest.main()
