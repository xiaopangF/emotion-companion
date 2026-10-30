import json, os, sys, threading, subprocess, tempfile, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'dev'))
sys.path.insert(0, str(P / 'participant'))
from data_split import holdout_indices
from evaluate import validate_row
import numpy as np

class IntegrationTests(unittest.TestCase):

    def test_duplicates_never_cross_training_split(self):
        samples = [{'id': f'{c}_t1', 'conversation_id': c, 'history': [{'role': 'user', 'content': text}]} for c, text in [('a', 'same'), ('b', 'same'), ('c', 'different')]]
        fit, tune, keys = holdout_indices(samples)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(0 in fit, 1 in fit)

    def test_official_python_entry_and_auxiliary(self):
        class Handler(BaseHTTPRequestHandler):

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prediction = {'emotion_label': 'joy', 'user_profile': {'personality_traits': [], 'interests': [], 'style': []}, 'response_text': '辛苦了，可以慢慢说。', 'memory_refs': []}
                payload = json.dumps({'choices': [{'message': {'content': json.dumps(prediction, ensure_ascii=False)}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory)
                model_dir = tmp / 'models'
                model_dir.mkdir()
                encoder = {'mode': 'users', 'vocab': {'你': 0}, 'idf': [1.0]}
                (model_dir / 'users_features.json').write_text(json.dumps(encoder), encoding='utf-8')
                (model_dir / 'style.json').write_text(json.dumps({'classes': ['', 'brief'], 'feature_mode': 'users'}), encoding='utf-8')
                np.savez_compressed(model_dir / 'style.npz', weight=np.zeros((1, 2), dtype=np.float32), bias=np.asarray([0.0, 1.0], dtype=np.float32))
                cfg = json.loads((P / 'participant/configs/inference_config.json').read_text())
                cfg.update(backend='openai_compatible', base_url=f'http://127.0.0.1:{server.server_port}/v1', api_model='fixture', auxiliary_model_dir=str(model_dir), auxiliary_fields=['style'])
                (tmp / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
                (tmp / 'keep.txt').write_text('keep')
                data = [{'id': f'test1_00000{i}_t1', 'history': [{'turn_id': 1, 'role': 'user', 'content': '最近压力很大，有点担心工作。'}]} for i in [1, 2, 3]]
                (tmp / 'input.jsonl').write_text('\n'.join((json.dumps(r, ensure_ascii=False) for r in data)), encoding='utf-8')
                env = {**os.environ, 'EVALUATION_CONFIG_PATH': str(tmp / 'config.json'), 'PYTHONIOENCODING': 'utf-8'}
                for name in ['EVALUATION_BACKEND', 'EVALUATION_BASE_URL', 'EVALUATION_API_MODEL', 'EVALUATION_MODEL_PATH']:
                    env.pop(name, None)
                result = subprocess.run([sys.executable, str(P / 'participant/run_inference.py'), str(tmp / 'input.jsonl'), str(tmp)], env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                predictions = [json.loads(s) for s in (tmp / 'submission.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual([r['id'] for r in predictions], [r['id'] for r in data])
                for row in predictions:
                    validate_row(row)
                    self.assertEqual(row['user_profile']['style'], ['brief'])
                    self.assertEqual(row['emotion_label'], 'joy')
                report = json.loads((tmp / 'performance_report.json').read_text())
                self.assertEqual(report['rounds'], 3)
                self.assertFalse(report['complete'])
                self.assertEqual(report['parse_failure_count'], 0)
                self.assertEqual((tmp / 'keep.txt').read_text(), 'keep')
        finally:
            server.shutdown()
            server.server_close()
if __name__ == '__main__':
    unittest.main(verbosity=2)
