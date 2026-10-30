import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNTIME = Path(__file__).resolve().parents[1] / "participant"


class OfficialFailureContractTests(unittest.TestCase):
    def test_bad_row_continues_and_result_directory_can_be_reused(self):
        requests = []
        valid = {"emotion_label": "joy", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "response_text": "听到这个消息很开心。", "memory_refs": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                content = "{}" if len(requests) <= 2 else json.dumps(valid, ensure_ascii=False)
                payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = root / "result"
                result.mkdir()
                preserved = {"keep.txt": b"KEEP", "invalid_output_diagnostics.jsonl": b"old diagnostics", "generator_predictions.jsonl": b"old predictions"}
                for name, content in preserved.items():
                    (result / name).write_bytes(content)
                inputs = [{"id": str(index), "history": [{"role": "user", "content": "你好"}]} for index in range(2)]
                (root / "input.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in inputs), encoding="utf8")
                config = {"backend": "openai_compatible", "base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_model": "fixture", "generation": {"max_new_tokens": 768, "do_sample": False}, "performance_rounds": 100, "hardware_label": "fixture", "invalid_output_retries": 1}
                (root / "config.json").write_text(json.dumps(config), encoding="utf8")
                environment = {key: value for key, value in os.environ.items() if not key.startswith("EVALUATION_")}
                environment.update(EVALUATION_CONFIG_PATH=str(root / "config.json"), PYTHONIOENCODING="utf8")
                command = [sys.executable, str(RUNTIME / "run_inference.py"), str(root / "input.jsonl"), str(result)]
                completed = subprocess.run(command, capture_output=True, text=True, encoding="utf8", env=environment, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                predictions = [json.loads(line) for line in (result / "submission.jsonl").read_text(encoding="utf8").splitlines()]
                self.assertEqual([row["id"] for row in predictions], ["0", "1"])
                self.assertEqual(predictions[0]["response_text"], "")
                self.assertEqual(predictions[1]["emotion_label"], "joy")
                performance = json.loads((result / "performance_report.json").read_text())
                self.assertEqual(performance["parse_failure_ids"], ["0"])
                self.assertEqual(performance["rounds"], 2)
                evidence = result / performance["diagnostics_directory"] / "invalid_output_diagnostics.jsonl"
                evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
                rerun = subprocess.run(command, capture_output=True, text=True, encoding="utf8", env=environment, timeout=30)
                self.assertEqual(rerun.returncode, 0, rerun.stderr)
                self.assertEqual(json.loads((result / "performance_report.json").read_text())["parse_failure_count"], 0)
                self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), evidence_hash)
                for name, content in preserved.items():
                    self.assertEqual((result / name).read_bytes(), content)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
