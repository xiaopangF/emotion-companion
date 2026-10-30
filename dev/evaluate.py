"""Auditable full-dataset metrics. Local definitions are not official score formulas."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'participant'))
from run_inference import EMOTIONS, PROFILE_LABELS
DIMS = tuple(PROFILE_LABELS)

def read_jsonl(path):
    out = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
    return out

def index_unique(rows):
    result = {}
    for row in rows:
        key = row.get('id')
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f'Empty, invalid, or duplicate ID: {key!r}')
        result[key] = row
    if not result:
        raise ValueError('Empty dataset')
    return result

def tokenize(text):
    return re.findall('[\\u4e00-\\u9fff]|[A-Za-z0-9]+|[^\\s]', text)

def corpus_bleu(references, hypotheses):
    """Single-reference corpus BLEU-4, unsmoothed, 0..100, explicit token lists."""
    matches = [0] * 4
    totals = [0] * 4
    ref_len = sum(map(len, references))
    hyp_len = sum(map(len, hypotheses))
    for ref, hyp in zip(references, hypotheses):
        for n in range(1, 5):
            hc = Counter((tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)))
            rc = Counter((tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)))
            matches[n - 1] += sum((hc & rc).values())
            totals[n - 1] += sum(hc.values())
    prec = [m / t if t else 0.0 for m, t in zip(matches, totals)]
    bp = math.exp(min(0, 1 - ref_len / hyp_len)) if hyp_len else 0.0
    score = 100 * bp * math.exp(sum((math.log(p) for p in prec)) / 4) if min(prec) > 0 else 0.0
    return {'score': score, 'brevity_penalty': bp, 'precisions': prec, 'reference_tokens': ref_len, 'hypothesis_tokens': hyp_len, 'smoothing': 'none', 'effective_order': False}

def lcs_len(a, b):
    """Exact bit-parallel LCS; no truncation or sample selection."""
    positions = {}
    for j, token in enumerate(b):
        positions[token] = positions.get(token, 0) | 1 << j
    row = 0
    for token in a:
        match = positions.get(token, 0) | row
        row = match & ~(match - (row << 1 | 1))
    return row.bit_count()

def rouge_l(refs, hyps):
    scores = []
    for ref, hyp in zip(refs, hyps):
        lcs = lcs_len(ref, hyp)
        scores.append(2 * lcs / (len(ref) + len(hyp)) if ref or hyp else 0.0)
    return 100 * statistics.fmean(scores)

def validate_row(row):
    required = {'id', 'response_text', 'emotion_label', 'user_profile', 'memory_refs'}
    if set(row) != required:
        raise ValueError(f"{row.get('id')}: output fields mismatch")
    if not isinstance(row['response_text'], str):
        raise ValueError('response_text must be string')
    if row['emotion_label'] not in EMOTIONS | {''}:
        raise ValueError('Invalid emotion')
    if not isinstance(row['user_profile'], dict) or set(row['user_profile']) != set(DIMS):
        raise ValueError('Invalid profile object')
    for dim in DIMS:
        values = row['user_profile'][dim]
        if not isinstance(values, list) or not all((isinstance(v, str) for v in values)) or len(set(values)) != len(values) or (not set(values) <= PROFILE_LABELS[dim]):
            raise ValueError(f'Invalid {dim}')
    refs = row['memory_refs']
    if not isinstance(refs, list) or not all((isinstance(v, str) and re.fullmatch('mem_[0-9]{6}', v) for v in refs)) or len(set(refs)) != len(refs):
        raise ValueError('Invalid memory refs')

def evaluate(gt_rows, pred_rows):
    gt = index_unique(gt_rows)
    pred = index_unique(pred_rows)
    if set(gt) != set(pred):
        raise ValueError(f'ID mismatch: missing={len(set(gt) - set(pred))} extra={len(set(pred) - set(gt))}')
    for p in pred.values():
        validate_row(p)
    keys = list(gt)
    targets = [gt[k].get('target', gt[k]) for k in keys]
    ps = [pred[k] for k in keys]
    emotions = sorted(EMOTIONS)
    confusion = {e: {p: 0 for p in emotions + ['']} for e in emotions}
    for t, p in zip(targets, ps):
        confusion[t['emotion_label']][p['emotion_label']] += 1
    per_class = {}
    for e in emotions:
        tp = confusion[e][e]
        support = sum(confusion[e].values())
        predicted = sum((confusion[t][e] for t in emotions))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[e] = {'support': support, 'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    profile = {}
    for d in DIMS:
        ts = [set(t['user_profile'][d]) for t in targets]
        qs = [set(p['user_profile'][d]) for p in ps]
        tp = sum((len(t & q) for t, q in zip(ts, qs)))
        fp = sum((len(q - t) for t, q in zip(ts, qs)))
        fn = sum((len(t - q) for t, q in zip(ts, qs)))
        profile[d] = {'exact_set_accuracy': sum((t == q for t, q in zip(ts, qs))) / len(ts), 'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, 'hamming_accuracy': 1 - (fp + fn) / (len(ts) * len(PROFILE_LABELS[d])), 'gold_empty_rate': sum((not t for t in ts)) / len(ts)}
    refs = [tokenize(t['response_text']) for t in targets]
    hyps = [tokenize(p['response_text']) for p in ps]
    accuracy = sum((t['emotion_label'] == p['emotion_label'] for t, p in zip(targets, ps))) / len(keys)
    pacc = statistics.fmean((profile[d]['exact_set_accuracy'] for d in DIMS))
    return {'samples': len(keys), 'emotion_accuracy': accuracy, 'emotion_macro_f1': statistics.fmean((v['f1'] for v in per_class.values())), 'per_class': per_class, 'confusion_matrix': confusion, 'profile': profile, 'profile_dimension_mean_exact_accuracy': pacc, 'bleu4': corpus_bleu(refs, hyps), 'rouge_l_f1': rouge_l(refs, hyps), 'bertscore': None, 'empty_response_count': sum((not p['response_text'].strip() for p in ps)), 'conditional_40_point_proxy': 25 * accuracy + 15 * pacc, 'metric_notes': {'scope': 'All examples; full responses; no truncation.', 'tokenization': 'Chinese characters, contiguous Latin/digits, punctuation tokens. Case preserved.', 'profile': 'Exact label-set match in each of three dimensions; official definition unconfirmed.', 'proxy': '25*emotion_accuracy + 15*mean_profile_exact_accuracy, only if organizer uses these linear formulas; NOT an official score.', 'official_total': 'Unavailable: response metric aggregation, BERTScore configuration, latency conversion and expert scores not supplied.'}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('--gt', type=Path, default=ROOT / 'dev/data/val_gt.jsonl')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--bertscore-model', help='Local encoder path, no implicit downloads')
    ap.add_argument('--bertscore-num-layers', type=int)
    ap.add_argument('--bertscore-device', default='cpu')
    ap.add_argument('--bertscore-batch-size', type=int, default=8)
    ap.add_argument('--performance-report', type=Path)
    a = ap.parse_args()
    gs = read_jsonl(a.gt)
    ps = read_jsonl(a.submission)
    result = evaluate(gs, ps)
    result['provenance'] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.gt, a.submission]}
    if a.performance_report:
        perf = json.loads(a.performance_report.read_text(encoding='utf-8'))
        if perf.get('submission_sha256') != hashlib.sha256(a.submission.read_bytes()).hexdigest():
            raise ValueError('Performance/submission hash mismatch')
        result['performance'] = perf
    if a.bertscore_model:
        if not Path(a.bertscore_model).is_dir() or a.bertscore_num_layers is None:
            raise ValueError('Local BERTScore model directory and --bertscore-num-layers required')
        from bert_score import score
        byid = index_unique(ps)
        pr, re, f1 = score([byid[g['id']]['response_text'] for g in gs], [g['target']['response_text'] for g in gs], model_type=a.bertscore_model, num_layers=a.bertscore_num_layers, device=a.bertscore_device, batch_size=a.bertscore_batch_size, rescale_with_baseline=False, verbose=True)
        result['bertscore'] = {'precision': pr.mean().item(), 'recall': re.mean().item(), 'f1': f1.mean().item(), 'model_path': a.bertscore_model, 'num_layers': a.bertscore_num_layers, 'rescale_with_baseline': False, 'official_configuration_confirmed': False}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + '\n', encoding='utf-8')
    print(text)
if __name__ == '__main__':
    main()
