import unittest, random
from tests.interface import Test
from scripts.metrics import compute_f1, compute_recall, normalized_inverse_class_frequency
from scripts.consts import CLASSIFICATION_METRICS


class MetricTest(Test):

    def generate_labels(self, size, num_labels, balanced=True):
        if balanced:
            return [random.choice(range(1, num_labels + 1)) for _ in range(size)]
        return [random.choices(range(1, num_labels + 1), weights=[i ** 2 for i in range(1, num_labels + 1)])[0] for _ in range(size)]


class ImportedMetricTest(MetricTest):
    """Validate the correctness of the functions imported from sklearn"""

    def manual_weighted_f1(self, y_true, y_pred):
        labels = set(y_true)
        label_counts = {label: y_true.count(label) for label in labels}
        
        weighted_f1 = 0.0
        for label in labels:
            f1 = compute_f1(y_true, y_pred, label)
            weighted_f1 += f1 * label_counts[label]
        
        weighted_f1 /= len(y_true)
        return weighted_f1

    def manual_balanced_accuracy(self, y_true, y_pred):
        labels = set(y_true)

        sum_recall = 0.0
        for label in labels:
            recall = compute_recall(y_true, y_pred, label)
            sum_recall += recall
        
        balanced_accuracy = sum_recall / len(labels)
        return balanced_accuracy
    
    def test_manual_weighted_f1(self):
        for _ in range(10):
            y_true = self.generate_labels(100, 5)
            y_pred = self.generate_labels(100, 5)
            self.assertAlmostEqual(self.manual_weighted_f1(y_true, y_pred), CLASSIFICATION_METRICS['f1_weighted'](y_true, y_pred), places=5)

    def test_manual_balanced_accuracy(self):
        for _ in range(10):
            y_true = self.generate_labels(100, 5)
            y_pred = self.generate_labels(100, 5)
            self.assertAlmostEqual(self.manual_balanced_accuracy(y_true, y_pred), CLASSIFICATION_METRICS['accuracy_balanced'](y_true, y_pred), places=5)


class ICFMetricTest(MetricTest):

    def test_basic_normalized_inverse_class_frequency(self):
        y_true = ['class1', 'class2', 'class1', 'class2', 'class2', 'class2']
        expected_output = {'class2': 2 / 6, 'class1': 4 / 6}
        self.assertEqual(normalized_inverse_class_frequency(y_true), expected_output)

    def test_advanced_normalized_inverse_class_frequency(self):
        y_true = self.generate_labels(100, 3, balanced=False)
        frequencies = normalized_inverse_class_frequency(y_true)
        self.assertLess(frequencies[2], frequencies[1])
        self.assertLess(frequencies[3], frequencies[2])

    def test_f1_weighted_icf_vs_f1_weighted(self):
        for _ in range(3):
            y_true = self.generate_labels(200, 3, balanced=False)
            y_pred = self.generate_labels(200, 3, balanced=False)
            self.assertLess(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), CLASSIFICATION_METRICS['f1_weighted'](y_true, y_pred))

    def test_multiclass_weighted_metric_using_icf(self):
        y_true = [0] * 5 + [1] * 90 + [2] * 5
        y_pred = [0] * 5 + [1] * 90 + [2] * 5
        self.assertEqual(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 1.0)

        y_true = [2] * 5 + [1] * 90 + [0] * 5
        y_pred = [0] * 5 + [1] * 90 + [2] * 5
        self.assertLess(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.1)

        y_true = [2] * 5 + [1] * 90 + [0] * 5
        y_pred = [1] * 100
        self.assertLess(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.5)

        y_true = [0] * 5 + [1] * 90 + [2] * 5
        y_pred = [0] * 1 + [1] * 98 + [2] * 1
        self.assertAlmostEqual(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.4, places=1)

        y_true = [0] * 1 + [1] * 98 + [2] * 1
        y_pred = [0] * 5 + [1] * 90 + [2] * 5
        self.assertAlmostEqual(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.3, places=1)

        y_true = [0] * 10 + [1] * 80 + [2] * 10
        y_pred = [0] * 7 + [1] * 86 + [2] * 7
        self.assertAlmostEqual(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.8, places=1)

        y_true = [0] * 7 + [1] * 86 + [2] * 7
        y_pred = [0] * 10 + [1] * 80 + [2] * 10
        self.assertAlmostEqual(CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred), 0.8, places=1)


class MetricReproducibilityTest(MetricTest):

    def test_metric_reproducibility(self):
        y_true = [1, 2, 3, 4, 4, 1, 2, 3, 4, 4]
        y_pred = [2, 4, 4, 1, 1, 1, 3, 1, 4, 1]
        result = CLASSIFICATION_METRICS['f1_weighted_icf'](y_true, y_pred)
        self.assertEqual(result, 0.12244897959183675)
        result = CLASSIFICATION_METRICS['recall_weighted_icf'](y_true, y_pred)
        self.assertEqual(result, 0.17857142857142855)


if __name__ == '__main__':
    unittest.main()
