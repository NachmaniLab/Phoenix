import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from scripts.consts import ALL_CELLS, SIZES, BackgroundMode
from tests.interface import Test
from scripts.utils import define_set_size, define_batch_size, order_gene_sets_by_size, convert_to_str, convert_from_str, enum2str, str2enum, remove_outliers, correct_effect_size, save_step_runtime, load_total_runtime, format_runtime, save_peak_memory, load_peak_memory, format_memory, _HAS_RESOURCE


class UtilTest(Test):

    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
        self.define_set_size = lambda set_len, set_fraction, min_set_size: define_set_size(set_len, set_fraction, min_set_size, all_sizes=SIZES)
    
    def test_set_size_definition(self):
        self.assertEqual(self.define_set_size(80, 0.25, 10), 20)
        self.assertEqual(self.define_set_size(100, 0.5, 1), 60)  # 50 would be exact, but 60 is the closest
        self.assertEqual(self.define_set_size(20, 0.5, 15), 15)  # min_set_size makes it 15 instead of 10
        self.assertEqual(self.define_set_size(20, 0.5, 16), 20)  # min_set_size makes it 16 instead of 10, but 20 is the closest
        self.assertEqual(self.define_set_size(20, 0.5, 40), 20)  # min_set_size turns 10 into 40 but 40 is bigger than set_len so set_len is selected
        self.assertEqual(self.define_set_size(8, 0.5, 10), 10)  # min_set_size turns 4 into 10 but 10 is bigger than set_len so set_len is selected, but since 8 is not in SIZES, 10 is selected
        self.assertEqual(self.define_set_size(2000, 0.5, 10), 500)  # as 1000 is not in SIZES
        self.assertEqual(self.define_set_size(600, 1.0, 10), 500)  # as 600 is not in SIZES
        self.assertEqual(self.define_set_size(600, 1.0, 550), 500)  # as 600 is not in SIZES and also no 550

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

    def test_save_and_load_peak_memory(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            save_peak_memory(tmp, 'step2', 512.0)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'memory_step2.txt')))
            self.assertAlmostEqual(load_peak_memory(tmp, 'step2'), 512.0)

    def test_save_peak_memory_with_batch(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            save_peak_memory(tmp, 'step2', 300.0, batch=1)
            save_peak_memory(tmp, 'step2', 500.0, batch=2)
            self.assertTrue(os.path.exists(os.path.join(tmp, 'memory_step2_batch1.txt')))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'memory_step2_batch2.txt')))
            self.assertAlmostEqual(load_peak_memory(tmp, 'step2'), 500.0)

    def test_load_peak_memory_returns_max(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            save_peak_memory(tmp, 'step2', 100.0, batch=1)
            save_peak_memory(tmp, 'step2', 999.0, batch=2)
            save_peak_memory(tmp, 'step2', 50.0, batch=3)
            self.assertAlmostEqual(load_peak_memory(tmp, 'step2'), 999.0)

    def test_load_peak_memory_all_steps(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            save_peak_memory(tmp, 'step1', 100.0)
            save_peak_memory(tmp, 'step2', 800.0)
            save_peak_memory(tmp, 'step3', 600.0)
            save_peak_memory(tmp, 'step4', 400.0)
            self.assertAlmostEqual(load_peak_memory(tmp), 800.0)

    def test_load_peak_memory_empty_dir(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertAlmostEqual(load_peak_memory(tmp, 'step2'), 0.0)

    def test_save_peak_memory_overwrites(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        with tempfile.TemporaryDirectory() as tmp:
            save_peak_memory(tmp, 'step1', 200.0)
            save_peak_memory(tmp, 'step1', 350.0)
            self.assertAlmostEqual(load_peak_memory(tmp, 'step1'), 350.0)

    def test_format_memory_mb(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        self.assertEqual(format_memory(0.0), '0.0 MB')
        self.assertEqual(format_memory(512.0), '512.0 MB')
        self.assertEqual(format_memory(1023.9), '1023.9 MB')

    def test_format_memory_gb(self):
        if not _HAS_RESOURCE:
            self.skipTest("Resource module not available, skipping memory tests.")
        self.assertEqual(format_memory(1024.0), '1.00 GB')
        self.assertEqual(format_memory(2048.0), '2.00 GB')
        self.assertEqual(format_memory(5263.36), '5.14 GB')

    def test_order_gene_sets_by_size(self):
        """Gene sets should be ordered from largest to smallest."""
        gene_sets = {'a': ['g1', 'g2', 'g3'], 'b': ['g4'], 'c': ['g5', 'g6']}
        result = order_gene_sets_by_size(gene_sets)
        self.assertEqual(list(result.keys()), ['a', 'c', 'b'])


if __name__ == '__main__':
    unittest.main()
