import os
import tempfile
import unittest
from scripts.consts import SIZES, BackgroundMode, LEN_SIZES
from scripts.backgrounds import (
    set_background_mode,
    define_sizes_in_random_mode,
    define_sizes_in_real_mode,
)
from scripts.output import load_background_scores, save_background_scores
from tests.interface import Test


class BackgroundModeTest(Test):

    def test_set_background_mode_auto_returns_real_when_enough_gene_sets(self):
        repeats = 10
        gene_set_len = repeats * LEN_SIZES  # exactly at threshold
        result = set_background_mode(BackgroundMode.AUTO, repeats, gene_set_len)
        self.assertEqual(result, BackgroundMode.REAL)

    def test_set_background_mode_auto_returns_random_when_not_enough_gene_sets(self):
        repeats = 10
        gene_set_len = repeats * LEN_SIZES - 1  # just below threshold
        result = set_background_mode(BackgroundMode.AUTO, repeats, gene_set_len)
        self.assertEqual(result, BackgroundMode.RANDOM)


class SizeDefinitionTest(Test):

    def setUp(self):
        self.gene_sets = {
            "gs1": list(range(10)),
            "gs2": list(range(25)),
            "gs3": list(range(40)),
            "gs4": list(range(90)),
            "gs5": list(range(130)),
            "gs6": list(range(500)),
            "gs7": list(range(700)),
        }
        self.repeats = 2

    def test_define_sizes_in_random_mode(self):
        sizes = define_sizes_in_random_mode(
            gene_sets=self.gene_sets,
            set_fraction=0.5,
            min_set_size=5,
        )
        assert len(sizes) <= len(SIZES)
        assert all(s in SIZES for s in sizes)
        assert sizes == [5, 10, 20, 40, 60, 200]

    def test_define_sizes_in_real_mode(self):
        sizes = define_sizes_in_real_mode(
            gene_sets=self.gene_sets,
            set_fraction=1.0,
            min_set_size=1,
            repeats=self.repeats,
        )
        assert len(sizes) == len(self.gene_sets) // self.repeats
        assert sizes == [(10 + 25) // 2, (40 + 90) // 2, 500]  # expected medians

    def test_raises_error_in_real_mode_when_not_enough_gene_sets(self):
        small_gene_sets = {
            'set1': ['gene1', 'gene2'],
            'set2': ['gene1', 'gene2', 'gene3'],
        }
        with self.assertRaises(RuntimeError) as context:
            define_sizes_in_real_mode(small_gene_sets, set_fraction=1.0, min_set_size=1)
        self.assertIn('Not enough gene sets', str(context.exception))


class TestBackgroundCacheIO(Test):

    def test_save_then_load_background_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            background = 'test-background'
            scores = [0.1, 0.2, 0.3]
            save_background_scores(scores, background, cache_path=tmp)
            loaded = load_background_scores(background, cache_path=tmp)
            self.assertEqual(loaded, scores)

    def test_load_raises_error_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as context:
                load_background_scores('does-not-exist', cache_path=tmp)
            self.assertIn('not found in cache', str(context.exception))

    def test_load_raises_error_when_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            background = "empty-file"
            open(os.path.join(tmp, f'{background}.yml'), "w").close()  # empty file
            with self.assertRaises(FileNotFoundError) as context:
                load_background_scores(background, cache_path=tmp)
            self.assertIn('not found in cache', str(context.exception))


if __name__ == '__main__':
    unittest.main()
