import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from prepare_corrections import prepare


class PrepareCorrectionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cv2.imwrite(str(self.root / 'clean.png'), np.full((4, 6, 3), 80, np.uint8))
        self.doc = {'schema': 'tooth-labels/1', 'cleanImage': 'clean.png', 'crop': [1, 1, 4, 2],
                    'width': 4, 'height': 2, 'labels': [[7, 2], [20, 2], [7, 2], [20, 2]]}
        (self.root / 'labels.json').write_text(json.dumps(self.doc))
        self.record = {'id': 'sample', 'group': 'person1', 'split': 'train', 'image': 'clean.png', 'labels': 'labels.json'}

    def run_prepare(self, records):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'samples': records}))
        return prepare(path, self.root / 'output')

    def test_instance_ids_and_native_geometry_survive(self):
        result = self.run_prepare([self.record])
        dataset = np.load(self.root / 'output/corrections_train.npz')
        self.assertEqual(dataset['img'].shape, (1, 120, 160, 3))
        self.assertEqual(set(np.unique(dataset['inst'])), {1, 2})
        self.assertTrue((dataset['inst'][0, :, :80] == 1).all())
        self.assertEqual(result['samples'][0]['crop'], [1, 1, 4, 2])
        self.assertEqual(result['counts'], {'train': 1, 'valid': 0, 'test': 0})

    def test_group_and_duplicate_image_leakage_rejected(self):
        for group in ['person1', 'different_name']:
            with self.assertRaisesRegex(ValueError, 'leakage'):
                self.run_prepare([self.record, {**self.record, 'id': 'second', 'group': group, 'split': 'test'}])

    def test_bad_label_runs_and_crop_rejected(self):
        for change in [{'labels': [[1, 9]]}, {'labels': [[1, 7]]}, {'crop': [5, 1, 4, 2]}]:
            (self.root / 'labels.json').write_text(json.dumps({**self.doc, **change}))
            with self.assertRaises(ValueError):
                self.run_prepare([self.record])

    def test_unreviewed_predictions_are_not_ground_truth(self):
        (self.root / 'labels.json').write_text(json.dumps({
            **self.doc, 'annotationStatus': 'unreviewed_model_draft'}))
        with self.assertRaisesRegex(ValueError, 'Review and correct'):
            self.run_prepare([self.record])


if __name__ == '__main__':
    unittest.main()
