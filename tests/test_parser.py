"""Regression coverage for truncated and invalid model-generated JSON."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'participant'))
from run_inference import parse_prediction, retry_config


class ParserTests(unittest.TestCase):
    def test_truncated_outer_object_does_not_become_nested_profile(self):
        raw = '{"emotion_label":"anxiety","user_profile":{"personality_traits":[],"interests":[],"style":[]},"response_text":"先练习一次'
        prediction, failed = parse_prediction('fixture', raw)
        self.assertTrue(failed)
        self.assertEqual(prediction['emotion_label'], 'anxiety')
        self.assertEqual(prediction['response_text'], '先练习一次')

    def test_invalid_fields_do_not_fabricate_labels(self):
        raw = {'emotion_label': 'unknown', 'response_text': '可以慢慢说。',
               'user_profile': {'personality_traits': ['fake'], 'interests': [], 'style': []}, 'memory_refs': []}
        prediction, failed = parse_prediction('fixture', json.dumps(raw))
        self.assertTrue(failed)
        self.assertEqual(prediction['emotion_label'], '')
        self.assertEqual(prediction['user_profile']['personality_traits'], [])
        self.assertEqual(prediction['response_text'], raw['response_text'])

    def test_retry_has_bounded_response_and_local_schema(self):
        config = retry_config({'api_options': {}})
        schema = config['api_options']['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['response_text']['maxLength'], 180)
        self.assertFalse(schema['additionalProperties'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
