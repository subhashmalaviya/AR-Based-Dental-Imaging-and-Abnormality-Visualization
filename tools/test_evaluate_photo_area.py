"""Small independent checks of evaluation arithmetic and ignore semantics."""
import unittest

import numpy as np

from evaluate_photo_area import region_mask, score


class PixelEvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        gt = np.array([[255, 255], [0, 0]], np.uint8)
        pred = np.array([[255, 0], [255, 0]], np.uint8)
        result = score(gt, pred, np.full((2, 2), 255, np.uint8))
        self.assertEqual([result[k] for k in ['tp', 'fp', 'fn', 'tn']], [1, 1, 1, 1])
        self.assertAlmostEqual(result['iou'], 1/3)
        for key in ['dice', 'precision', 'recall', 'pixel_accuracy']:
            self.assertEqual(result[key], 0.5)

    def test_ignored_pixel_is_not_a_true_negative(self):
        gt = np.array([[255, 0]], np.uint8)
        pred = np.array([[255, 255]], np.uint8)
        result = score(gt, pred, np.array([[255, 0]], np.uint8))
        self.assertEqual(result['evaluated_pixels'], 1)
        self.assertEqual(result['tn'], 0)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['iou'], 1)

    def test_no_prediction_keeps_false_negatives(self):
        result = score(np.array([[255]], np.uint8), np.array([[0]], np.uint8),
                       np.array([[255]], np.uint8))
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['iou'], 0)
        self.assertEqual(result['recall'], 0)
        self.assertIsNone(result['precision'])

    def test_rectangles_use_xy_width_height(self):
        mask = region_mask((8, 10), [(2, 3, 4, 2)])
        self.assertEqual(np.count_nonzero(mask), 8)
        self.assertEqual(mask[3, 2], 255)
        self.assertEqual(mask[4, 5], 255)
        self.assertEqual(mask[5, 5], 0)
        self.assertEqual(mask[4, 6], 0)


if __name__ == '__main__':
    unittest.main()
