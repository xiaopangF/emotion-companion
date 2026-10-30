"""Check image layout, artifacts and official entry point without changing model weights."""
import argparse, hashlib, json, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participant', type=Path, default=Path(__file__).resolve().parents[1] / 'participant')
    ap.add_argument('--result-dir', type=Path)
    ap.add_argument('--strict-platform', action='store_true')
    ap.add_argument('--team-name')
    a = ap.parse_args()
    p = a.participant.resolve()
    checks = {}
    checks['entrypoint_exists'] = (p / 'start.sh').is_file()
    checks['runner_exists'] = (p / 'run_inference.py').is_file()
    cfg = json.loads((p / 'configs/inference_config.json').read_text(encoding='utf-8-sig'))
    model = Path(cfg['model_path'])
    model = model if model.is_absolute() else p / model
    checks['model_config'] = (model / 'config.json').is_file()
    checks['model_weights'] = model.is_dir() and (any(model.glob('*.safetensors')) or any(model.glob('pytorch_model*.bin')))
    checks['tokenizer'] = any(((model / n).is_file() for n in ['tokenizer.json', 'tokenizer.model', 'vocab.json']))
    if cfg.get('auxiliary_model_dir'):
        aux = Path(cfg['auxiliary_model_dir'])
        aux = aux if aux.is_absolute() else p / aux
        for field in cfg.get('auxiliary_fields', []):
            checks['aux_' + field] = all(((aux / (field + ext)).exists() for ext in ['.json', '.npz']))
    if a.result_dir:
        sub = a.result_dir / 'submission.jsonl'
        perf = a.result_dir / 'performance_report.json'
        checks['submission_exists'] = sub.exists()
        checks['performance_exists'] = perf.exists()
        if sub.exists() and perf.exists():
            report = json.loads(perf.read_text(encoding='utf-8'))
            checks['matching_hash'] = report['submission_sha256'] == hashlib.sha256(sub.read_bytes()).hexdigest()
            checks['100_rounds'] = report['rounds'] == 100
            checks['no_parse_failures'] = report['parse_failure_count'] == 0
    if a.strict_platform:
        checks['official_path'] = p == Path('/root/participant')
        checks['team_name_supplied'] = bool(a.team_name)
        if a.team_name:
            checks['expert_material_directory'] = (Path('/root') / f'AI技术赛道-数字人综合情感对话模型-{a.team_name}').is_dir()
    print(json.dumps({'checks': checks, 'ready': all(checks.values()), 'meaning': 'No training or official score is implied.'}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)
if __name__ == '__main__':
    main()
