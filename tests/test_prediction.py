import unittest
import time
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from tests.interface import Test
from scripts.prediction import create_cv, get_train_target, get_train_data, train, compare_scores, get_prediction_score, encode_labels
from scripts.step_2_pathway_scoring import get_gene_set_batch
from scripts.utils import adjust_p_value
from scripts.consts import CELL_TYPE_COL, ALL_CELLS, CLASSIFICATION_PREDICTOR, CLASSIFICATION_PREDICTOR_ARGS, METRICS, REGRESSION_PREDICTOR, REGRESSION_PREDICTOR_ARGS, THRESHOLD, CLASSIFICATION_METRIC, REGRESSION_METRIC, FEATURE_SELECTION, SEED


class LabelEncodingTest(Test):
    def test_label_encoding(self):
        string_y = pd.Series(['TypeA', 'TypeB', 'TypeA', 'TypeC', 'TypeB', 'TypeC'])
        numeric_y = encode_labels(string_y)
        self.assertEqual(numeric_y.tolist(), [0, 1, 0, 2, 1, 2])

        boolean_y = pd.Series([True, False, True, False, True])
        numeric_y = encode_labels(boolean_y)
        self.assertEqual(numeric_y.tolist(), [1, 0, 1, 0, 1])


class TrainDataTest(Test):
    def setUp(self):

        self.scaled_expression = pd.DataFrame({
            'Gene1': [1.0, 1.3, 1.5, 1.7],  # similar across cell types
            'Gene2': [10, 1, 10, 1],  # very different across cell types
            'Gene3': [3, 3, 3, 5],  # a little different across cell types
            'Gene4': [4, 4, 4, 4],  # identical across cell types
            'Gene5': [10, 1, 10, 1],  # very different across cell types
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        self.scaled_pseudotime = pd.DataFrame({
            1: [0.1, 0.2, 0.3, None],
            2: [0.6, 0.3, 0.9, 0.1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

    def test_cell_type_target(self):
        # Test case for get_train_target function with cell_types
        cell_type = 'TypeB'
        cell_type_target = get_train_target(cell_types=self.cell_types, cell_type=cell_type)
        self.assertEqual(cell_type_target.tolist(), [c == cell_type for c in self.cell_types[CELL_TYPE_COL]])

        cell_type_target = get_train_target(cell_types=self.cell_types, cell_type=ALL_CELLS)
        self.assertEqual(cell_type_target.tolist(), self.cell_types[CELL_TYPE_COL].tolist())

    def test_pseudotime_target(self):
        # Test case for get_train_target function with pseudotime
        pseudotime_target = get_train_target(scaled_pseudotime=self.scaled_pseudotime, lineage=2)
        self.assertEqual(pseudotime_target.tolist(), self.scaled_pseudotime[2].tolist())

    def test_data_without_selection(self):
        # Test case where using cell_types and cell_type
        X, _, features, importances = get_train_data(
            self.scaled_expression,
            features=self.scaled_expression.columns, 
            cell_types=self.cell_types,
            cell_type='TypeA',
            set_size=None,
            feature_selection=None
        )

        self.assertEqual(X.shape, self.scaled_expression.shape)
        self.assertListEqual(sorted(features), sorted(self.scaled_expression.columns.tolist()))
        self.assertIsNone(importances)

    def test_cell_type_data_with_anova(self):
        # Test case where using cell_types and cell_type
        cell_dim = self.scaled_expression.shape[0]

        for cell_type in [ALL_CELLS, 'TypeA', 'TypeB']:

            set_size = 4
            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=self.scaled_expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)
            self.assertTrue('Gene4' not in selected_genes)

            set_size = 3
            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=['Gene1', 'Gene3', 'Gene2', 'Gene4', 'Gene5'], 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)
            self.assertEqual(selected_genes, ['Gene3', 'Gene2', 'Gene5']) # SelectKBest does not change order of features

            set_size = 2
            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=self.scaled_expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)
            self.assertEqual(selected_genes, ['Gene2', 'Gene5']) # SelectKBest does not change order of features

            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=['Gene1', 'Gene3', 'Gene5'],  # missing good predictor Gene2
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)
            self.assertEqual(selected_genes, ['Gene3', 'Gene5']) # SelectKBest does not change order of features
            
    def test_cell_type_data_with_rf(self):
        # Test case where using cell_types and cell_type
        cell_dim = self.scaled_expression.shape[0]

        for cell_type in [ALL_CELLS, 'TypeA', 'TypeB']:

            set_size = self.scaled_expression.shape[1] - 1
            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=self.scaled_expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='RF'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)

            # RandomForestClassifier orders features by importance:
            self.assertTrue(importances == sorted(importances, reverse=True))
            self.assertTrue(selected_genes[0] in ['Gene5', 'Gene2']) 
            self.assertTrue(selected_genes[1] in ['Gene5', 'Gene2'])
            self.assertTrue('Gene4' not in selected_genes) 

            set_size = self.scaled_expression.shape[1] - 2
            X, y, selected_genes, importances = get_train_data(
                self.scaled_expression,
                features=[g for g in self.scaled_expression.columns if g != 'Gene2'], 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='RF'
            )

            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(len(importances), set_size)
            self.assertTrue(importances == sorted(importances, reverse=True))
            self.assertTrue(selected_genes[0] == 'Gene5') 
            self.assertTrue('Gene4' not in selected_genes) 

    def test_pseudotime_data_with_anova(self):
        # Test case where using pseudotime and lineage

        lineage = 1
        cell_dim = self.scaled_pseudotime[lineage].dropna().shape[0]
        set_size = 1
        X, y, selected_genes, importances = get_train_data(
            scaled_expression=self.scaled_expression,
            features=self.scaled_expression.columns,
            scaled_pseudotime=self.scaled_pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='ANOVA'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(len(importances), set_size)
        self.assertTrue(selected_genes[0] == 'Gene1')  # Gene1 increases across time in lineage 1

        lineage = 2
        cell_dim = self.scaled_pseudotime[lineage].dropna().shape[0]
        set_size = 3
        X, y, selected_genes, importances = get_train_data(
            scaled_expression=self.scaled_expression,
            features=self.scaled_expression.columns,
            scaled_pseudotime=self.scaled_pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='ANOVA'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(len(importances), set_size)
        self.assertEqual(selected_genes, ['Gene2', 'Gene3', 'Gene5'])  # these 3 genes increase / decrease according to pseudotime order (and SelectKBest does not change order of features)

    def test_pseudotime_data_with_rf(self):
        # Test case where using pseudotime and lineage

        set_size = self.scaled_expression.shape[1] - 1

        lineage = 1
        cell_dim = self.scaled_pseudotime[lineage].dropna().shape[0]
        X, y, selected_genes, importances = get_train_data(
            scaled_expression=self.scaled_expression,
            features=self.scaled_expression.columns,
            scaled_pseudotime=self.scaled_pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='RF'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(len(importances), set_size)
        self.assertTrue(importances == sorted(importances, reverse=True))
        self.assertTrue(selected_genes[0] == 'Gene1')  # Gene1 increases across time in lineage 1

        lineage = 2
        cell_dim = self.scaled_pseudotime[lineage].dropna().shape[0]
        X, y, selected_genes, importances = get_train_data(
            scaled_expression=self.scaled_expression,
            features=self.scaled_expression.columns,
            scaled_pseudotime=self.scaled_pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='RF'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(len(importances), set_size)
        self.assertTrue(importances == sorted(importances, reverse=True))
        self.assertEqual(selected_genes[0], 'Gene1')  # changes across each pseudotime step
        self.assertTrue('Gene4' not in selected_genes)  # does not change across pseudotime order

    def test_data_with_full_size_selection(self):
        X, _, features, importances = get_train_data(
            self.scaled_expression,
            features=self.scaled_expression.columns, 
            cell_types=self.cell_types,
            cell_type='TypeA',
            set_size=self.scaled_expression.shape[1] + 100,  # more
            feature_selection='RF'
        )

        self.assertEqual(X.shape, self.scaled_expression.shape)
        self.assertListEqual(sorted(features), sorted(self.scaled_expression.columns.tolist()))  # RF sorts features by importance
        self.assertTrue(importances == sorted(importances, reverse=True))
        self.assertEqual(len(importances), self.scaled_expression.shape[1])

        X, _, features, importances = get_train_data(
            self.scaled_expression,
            features=self.scaled_expression.columns, 
            cell_types=self.cell_types,
            cell_type='TypeA',
            set_size=self.scaled_expression.shape[1] + 100,  # more
            feature_selection='ANOVA'
        )

        self.assertEqual(X.shape, self.scaled_expression.shape)
        self.assertListEqual(features, self.scaled_expression.columns.tolist())  # SelectKBest does not change order of features
        self.assertEqual(len(importances), self.scaled_expression.shape[1])


class TrainingPerformanceTest(Test):
    def setUp(self):

        self.scaled_expression = pd.DataFrame({
            'Gene1': [1.0, 1.3, 1.5, 1.7],  # similar across cell types
            'Gene2': [10, 1, 10, 1],  # very different across cell types
            'Gene3': [3, 3, 3, 5],  # a little different across cell types
            'Gene4': [4, 4, 4, 4],  # identical across cell types
            'Gene5': [10, 1, 10, 1],  # very different across cell types
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.cell_type_target = encode_labels(get_train_target(cell_types=cell_types, cell_type=ALL_CELLS))

        scaled_pseudotime = pd.DataFrame({
            1: [0.1, 0.2, 0.3, 0.4],
            2: [0.6, 0.3, 0.9, 0.1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.pseudotime_target = get_train_target(scaled_pseudotime=scaled_pseudotime, lineage=1)
        self.cross_validation = 2

        self.run_times = 10

    def test_classification_training_performance(self):
        X = np.array(self.scaled_expression[['Gene2', 'Gene5']])
        predictor = CLASSIFICATION_PREDICTOR
        predictor_args = CLASSIFICATION_PREDICTOR_ARGS
        model = predictor(**predictor_args)
        score_function = make_scorer(METRICS['f1_weighted_icf'], greater_is_better=True)
        cv = create_cv(is_regression=False, n_splits=self.cross_validation)
        
        durations = []
        for _ in range(self.run_times):
            start = time.time()
            train(
                X=X,
                y=self.cell_type_target,
                model=model,
                score_function=score_function,
                cv=cv
            )
            end = time.time()
            duration = end - start
            durations.append(duration)
        
        mean_duration = sum(durations) / len(durations)
        self.assertLessEqual(mean_duration, 0.2)

    def test_regression_training_performance(self):
        X = np.array(self.scaled_expression[['Gene1']])
        predictor = REGRESSION_PREDICTOR
        predictor_args = REGRESSION_PREDICTOR_ARGS
        model = predictor(**predictor_args)
        score_function = make_scorer(METRICS['neg_mean_squared_error'], greater_is_better=True)
        cv = create_cv(is_regression=True, n_splits=self.cross_validation)

        durations = []
        for _ in range(self.run_times):
            start = time.time()
            train(
                X=X,
                y=self.pseudotime_target,
                model=model,
                score_function=score_function,
                cv=cv
            )
            end = time.time()
            duration = end - start
            durations.append(duration)

        mean_duration = sum(durations) / len(durations)
        self.assertLessEqual(mean_duration, 0.2)


class TrainingTest(Test):
    def setUp(self):

        self.scaled_expression = pd.DataFrame({
            'Gene1': [1.0, 1.3, 1.5, 1.7],  # similar across cell types
            'Gene2': [10, 1, 10, 1],  # very different across cell types
            'Gene3': [3, 3, 3, 5],  # a little different across cell types
            'Gene4': [4, 4, 4, 4],  # identical across cell types
            'Gene5': [10, 1, 10, 1],  # very different across cell types
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.cell_type_target = encode_labels(get_train_target(cell_types=cell_types, cell_type=ALL_CELLS))

        scaled_pseudotime = pd.DataFrame({
            1: [0.1, 0.2, 0.3, 0.4],
            2: [0.6, 0.3, 0.9, 0.1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.pseudotime1_target = get_train_target(scaled_pseudotime=scaled_pseudotime, lineage=1)
        self.pseudotime2_target = get_train_target(scaled_pseudotime=scaled_pseudotime, lineage=2)

        self.cross_validation = 2
    
    def test_classification_training(self):
        cv = create_cv(is_regression=False, n_splits=self.cross_validation)

        good_features = np.array(self.scaled_expression[['Gene2', 'Gene5']])
        middle_features = np.array(self.scaled_expression[['Gene2', 'Gene1']])
        bad_features = np.array(self.scaled_expression[['Gene4', 'Gene1']])
        
        for metric in ['f1_weighted_icf', 'accuracy_balanced']:

                predictor = CLASSIFICATION_PREDICTOR
                predictor_args = CLASSIFICATION_PREDICTOR_ARGS
                model = predictor(**predictor_args)
                score_function = make_scorer(METRICS[metric], greater_is_better=True)
                
                good_score = train(good_features, self.cell_type_target, model, score_function, cv)
                middle_score = train(middle_features, self.cell_type_target, model, score_function, cv)
                bad_score = train(bad_features, self.cell_type_target, model, score_function, cv)

                self.assertEqual(good_score, 1)
                self.assertGreaterEqual(good_score, middle_score)
                self.assertGreaterEqual(middle_score, bad_score)

    def test_regression_training(self):
        cv = create_cv(is_regression=True, n_splits=self.cross_validation)

        # Lineage 1
        good_features = np.array(self.scaled_expression[['Gene1']])
        good_features2 = np.array(self.scaled_expression[['Gene3']])
        bad_features = np.array(self.scaled_expression[['Gene4']])
        
        predictor = REGRESSION_PREDICTOR
        predictor_args = REGRESSION_PREDICTOR_ARGS
        for metric in ['neg_mean_squared_error']:

                model = predictor(**predictor_args)
                score_function = make_scorer(METRICS[metric], greater_is_better=True)
                                
                good_score = train(good_features, self.pseudotime1_target, model, score_function, cv)
                good_score2 = train(good_features2, self.pseudotime1_target, model, score_function, cv)
                bad_score = train(bad_features, self.pseudotime1_target, model, score_function, cv)

                self.assertGreaterEqual(good_score, bad_score)
                self.assertGreaterEqual(good_score2, bad_score)

        # Lineage 2
        good_features = np.array(self.scaled_expression[['Gene2']])
        good_features2 = np.array(self.scaled_expression[['Gene5']])
        bad_features = np.array(self.scaled_expression[['Gene4']])
        
        for metric in ['neg_mean_squared_error']:

                model = predictor(**predictor_args)
                score_function = make_scorer(METRICS[metric], greater_is_better=True)
                
                good_score = train(good_features, self.pseudotime2_target, model, score_function, cv)
                good_score2 = train(good_features2, self.pseudotime2_target, model, score_function, cv)
                bad_score = train(bad_features, self.pseudotime2_target, model, score_function, cv)

                self.assertGreaterEqual(good_score, bad_score)
                self.assertGreaterEqual(good_score2, bad_score)


class ScoreComparisonTest(Test):
    def setUp(self):
        self.pos_background_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.neg_background_scores = [-i for i in self.pos_background_scores]
    
    def test_score_comparison(self):
        assert compare_scores(pathway_score=0.7, background_scores=self.pos_background_scores, distribution='normal') <= THRESHOLD
        assert compare_scores(pathway_score=0.2, background_scores=self.pos_background_scores, distribution='normal') > THRESHOLD

        assert compare_scores(pathway_score=0.7, background_scores=self.pos_background_scores, distribution='gamma') <= THRESHOLD
        assert compare_scores(pathway_score=0.2, background_scores=self.pos_background_scores, distribution='gamma') > THRESHOLD

        assert compare_scores(pathway_score=0.7, background_scores=self.pos_background_scores, distribution='normal') <= compare_scores(pathway_score=0.7, background_scores=self.pos_background_scores, distribution='gamma')

    def test_neg_score_comparison(self):
        assert compare_scores(pathway_score=-0.07, background_scores=self.neg_background_scores, distribution='normal') <= THRESHOLD
        assert compare_scores(pathway_score=-0.5, background_scores=self.neg_background_scores, distribution='normal') > THRESHOLD

        assert compare_scores(pathway_score=-0.01, background_scores=self.neg_background_scores, distribution='gamma') <= THRESHOLD
        assert compare_scores(pathway_score=-0.5, background_scores=self.neg_background_scores, distribution='gamma') > THRESHOLD

        assert compare_scores(pathway_score=-0.07, background_scores=self.neg_background_scores, distribution='normal') <= compare_scores(pathway_score=-0.07, background_scores=self.neg_background_scores, distribution='gamma')

    def test_multiple_corrections(self):
        adjust_p_value([0.0, 0.05, 0.5, 1.0])


class PredictionScoreTest(Test):
    def setUp(self) -> None:

        self.scaled_expression = pd.DataFrame({
            'Gene1': [1, 3, 5, 7, 9, 11],
            'Gene2': [10, 1, 10, 1, 10, 1],
            'Gene3': [3, 4, 4, 5, 6, 7],
            'Gene4': [4, 4, 4, 4, 4, 4],
            'Gene5': [10, 1, 10, 1, 10, 1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.scaled_pseudotime = pd.DataFrame({
            1: [0.1, 0.25, 0.3, None, 0.44, 0.5],
            2: [0.6, 0.3, 0.9, 0.1, 1.0, 1.2],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.cross_validation = 3

    def _get_p_value(self, gene_set, background_scores, **kwargs):
        pathway_score = get_prediction_score(gene_set=gene_set, seed=SEED, feature_selection=FEATURE_SELECTION, **kwargs)[0]
        return compare_scores(pathway_score, background_scores, distribution='gamma')

    def _generate_background(self, repeats: int, **kwargs):
        """Generate background scores using random gene sets."""
        background_scores = []
        for i in range(repeats):
            score = get_prediction_score(seed=i, **kwargs)[0]
            background_scores.append(score)
        return background_scores

    def test_classification_significance(self):
        predictor = CLASSIFICATION_PREDICTOR
        base_args = {
            'scaled_expression': self.scaled_expression,
            'predictor': predictor,
            'predictor_args': CLASSIFICATION_PREDICTOR_ARGS,
            'set_size': 1,
            'score_function': make_scorer(METRICS[CLASSIFICATION_METRIC], greater_is_better=True),
            'cv': create_cv(is_regression=False, n_splits=self.cross_validation),
            'cell_types': self.cell_types,
            'cell_type': ALL_CELLS,
        }

        # Generate background once
        background = self._generate_background(repeats=50, **base_args)

        # Good gene sets should be significant
        p_value = self._get_p_value(['Gene2'], background, **base_args)
        self.assertLessEqual(p_value, THRESHOLD)
        p_value = self._get_p_value(['Gene5'], background, **base_args)
        self.assertLessEqual(p_value, THRESHOLD)

        # Bad gene sets should NOT be significant
        p_value = self._get_p_value(['Gene4'], background, **base_args)
        self.assertGreaterEqual(p_value, THRESHOLD)
        p_value = self._get_p_value(['Gene1'], background, **base_args)
        self.assertGreaterEqual(p_value, THRESHOLD)

    def test_regression_significance(self):
        predictor = REGRESSION_PREDICTOR
        base_args = {
            'scaled_expression': self.scaled_expression,
            'predictor': predictor,
            'predictor_args': REGRESSION_PREDICTOR_ARGS,
            'set_size': 1,
            'score_function': make_scorer(METRICS[REGRESSION_METRIC], greater_is_better=True),
            'cv': create_cv(is_regression=True, n_splits=self.cross_validation),
            'scaled_pseudotime': self.scaled_pseudotime,
            'lineage': 1,
        }

        # Generate background once
        background = self._generate_background(repeats=10, **base_args)

        # Good gene sets should be significant
        p_value = self._get_p_value(['Gene1'], background, **base_args)
        self.assertLessEqual(p_value, THRESHOLD)
        p_value = self._get_p_value(['Gene3'], background, **base_args)
        self.assertLessEqual(p_value, THRESHOLD)

        # Bad gene sets should NOT be significant
        p_value = self._get_p_value(['Gene2'], background, **base_args)
        self.assertGreaterEqual(p_value, THRESHOLD)
        p_value = self._get_p_value(['Gene4'], background, **base_args)
        self.assertGreaterEqual(p_value, THRESHOLD)
  

class GetBatchTest(Test):
    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
                    
    def test_get_gene_set_batch(self):
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=0, batch_size=-1), self.gene_sets)
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=3), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=2, batch_size=3), {'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=3, batch_size=2), {'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=4), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4']})


if __name__ == '__main__':
    unittest.main()
