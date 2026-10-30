"""Synthetic data and CLI checks without training data or model downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DataPreparationTests(unittest.TestCase):
    def test_future_turns_are_excluded_from_training_input(self):
        target = {"emotion_label": "joy", "user_profile": {"personality_traits": [], "interests": [], "style": []},
                  "response_text": "今天的努力值得肯定。", "memory_refs": []}
        conversation = {"conversation_id": "fixture", "turns": [
            {"turn_id": 1, "role": "user", "content": "终于完成了！"},
            {"turn_id": 2, "role": "assistant", "content": "FUTURE_CONTENT_NOT_IN_INPUT"}],
            "prediction_points": [{"target_user_turn_id": 1, "target": target}]}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for split in ('train', 'val'):
                folder = temporary / 'source' / split
                folder.mkdir(parents=True)
                (folder / f'{split}_public.jsonl').write_text(json.dumps(conversation, ensure_ascii=False) + '\n', encoding='utf8')
            environment = {**os.environ, 'PYTHONIOENCODING': 'utf8'}
            result = subprocess.run([sys.executable, str(ROOT / 'dev/prepare_data.py'), '--data-root', str(temporary / 'source'),
                                     '--output-dir', str(temporary / 'prepared')], env=environment, capture_output=True, text=True, encoding='utf8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            sample = json.loads((temporary / 'prepared/train_test.jsonl').read_text(encoding='utf8'))
            self.assertEqual(len(sample['history']), 1)
            formatted = json.loads((temporary / 'prepared/train_sft.jsonl').read_text(encoding='utf8'))
            self.assertNotIn('FUTURE_CONTENT_NOT_IN_INPUT', formatted['messages'][1]['content'])
            self.assertEqual(json.loads(formatted['messages'][-1]['content']), target)

    def test_adapter_initialization_cannot_resume_optimizer_simultaneously(self):
        result = subprocess.run([sys.executable, str(ROOT / 'dev/train_sft.py'), '--model', 'fixture',
                                 '--output-dir', 'unused', '--init-adapter', 'fixture', '--resume-from-checkpoint', 'fixture'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('mutually exclusive', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
