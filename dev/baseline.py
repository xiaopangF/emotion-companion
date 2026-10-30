"""Pure NumPy character n-gram baseline; train-only tuning, no pretrained downloads."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'participant'))
from run_inference import EMOTIONS, PROFILE_LABELS
DIMS = tuple(PROFILE_LABELS)

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def write_rows(path, data):
    with Path(path).open('w', encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')

def text_for(sample, mode):
    hist = sample['history']
    users = [t['content'] for t in hist if t['role'] == 'user']
    if mode == 'latest':
        return users[-1] if users else ''
    return '\n'.join(users)[-6000:]

def grams(text):
    text = ''.join(text.lower().split())
    return Counter((text[i:i + n] for n in (1, 2, 3) for i in range(len(text) - n + 1)))

def split_key(sample):
    return sample.get('conversation_id') or sample['id'].rsplit('_t', 1)[0]

def tune_split(sample):
    return int(hashlib.sha256(split_key(sample).encode()).hexdigest()[:8], 16) % 10 == 0

class NBModel:

    def __init__(self, classes, vocab, alpha=0.5, prior_weight=1.0):
        self.classes = classes
        self.vocab = vocab
        self.alpha = alpha
        self.prior_weight = prior_weight

    def fit(self, features, labels):
        counts = np.zeros((len(self.classes), len(self.vocab)), dtype=np.float64)
        cc = np.zeros(len(self.classes))
        idx = {s: i for i, s in enumerate(self.classes)}
        for feat, label in zip(features, labels):
            c = idx[label]
            cc[c] += 1
            for term, n in feat.items():
                j = self.vocab.get(term)
                if j is not None:
                    counts[c, j] += min(n, 3)
        self.counts = counts
        self.class_counts = cc
        self.set_alpha(self.alpha)
        return self

    def set_alpha(self, alpha):
        self.alpha = alpha
        totals = self.counts + self.alpha
        self.logprobs = np.log(totals / totals.sum(axis=1, keepdims=True)).astype(np.float32)
        self.priors = (np.log((self.class_counts + 1) / (self.class_counts.sum() + len(self.classes))) * self.prior_weight).astype(np.float32)

    def scores(self, feat):
        pairs = [(self.vocab[g], min(n, 3)) for g, n in feat.items() if g in self.vocab]
        if not pairs:
            return self.priors.copy()
        ids, counts = zip(*pairs)
        return self.logprobs[:, ids] @ np.asarray(counts, dtype=np.float32) + self.priors

    def predict(self, feat):
        return self.classes[int(np.argmax(self.scores(feat)))]

    def save(self, directory, name):
        np.savez_compressed(directory / (name + '.npz'), logprobs=self.logprobs, priors=self.priors)
        (directory / (name + '.json')).write_text(json.dumps({'classes': self.classes, 'vocab': self.vocab, 'alpha': self.alpha, 'prior_weight': self.prior_weight}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, directory, name):
        meta = json.loads((directory / (name + '.json')).read_text(encoding='utf-8'))
        obj = cls(**meta)
        z = np.load(directory / (name + '.npz'), allow_pickle=False)
        obj.logprobs = z['logprobs']
        obj.priors = z['priors']
        return obj

def vocabulary(features, limit):
    df = Counter()
    for f in features:
        df.update(f.keys())
    common = [g for g, c in df.most_common(limit) if c >= 2]
    return {g: i for i, g in enumerate(common)}

class Retrieval:
    """Retrieves responses from public training examples only, using input text."""

    def __init__(self, samples, targets):
        self.responses = [t['target']['response_text'] for t in targets]
        self.ids = [s['id'] for s in samples]
        self.postings = {}
        self.norms = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(samples):
            terms = set(grams(text_for(s, 'latest')))
            for g in terms:
                self.postings.setdefault(g, []).append(i)
        total = len(samples)
        self.idf = {g: math.log(1 + total / len(ids)) for g, ids in self.postings.items() if len(g) > 1 and 2 <= len(ids) <= total * 0.2}
        self.postings = {g: np.asarray(self.postings[g], dtype=np.int32) for g in self.idf}
        for g, ids in self.postings.items():
            self.norms[ids] += self.idf[g] ** 2
        self.norms = np.sqrt(np.maximum(self.norms, 1.0))

    def predict(self, sample):
        terms = sorted((g for g in set(grams(text_for(sample, 'latest'))) if g in self.idf), key=lambda g: self.idf[g], reverse=True)[:160]
        scores = np.zeros(len(self.responses), dtype=np.float32)
        for g in terms:
            scores[self.postings[g]] += self.idf[g] ** 2
        scores /= self.norms
        return self.responses[int(np.argmax(scores))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'dev/data')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'runs/char_baseline')
    ap.add_argument('--vocab-size', type=int, default=50000)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = rows(a.data_dir / 'train_test.jsonl')
    truth = rows(a.data_dir / 'train_gt.jsonl')
    val = rows(a.data_dir / 'val_test.jsonl')
    val_gt = rows(a.data_dir / 'val_gt.jsonl')
    train_lookup = {r['id']: r for r in truth}
    truth = [train_lookup[s['id']] for s in train]
    from data_split import holdout_indices
    fit_idx, tune_idx, group_ids = holdout_indices(train)
    assert {split_key(train[i]) for i in fit_idx}.isdisjoint({split_key(train[i]) for i in tune_idx})
    print(f'TRAIN={len(train)} FIT={len(fit_idx)} TUNE={len(tune_idx)} VAL={len(val)}', flush=True)
    model_dir = a.output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    models = {}
    tuning = []
    for mode, dims in [('latest', ['emotion_label']), ('users', list(DIMS))]:
        features = [grams(text_for(s, mode)) for s in train]
        vocab = vocabulary([features[i] for i in fit_idx], a.vocab_size)
        for dim in dims:
            labels = [t['target']['emotion_label'] if dim == 'emotion_label' else '|'.join(sorted(t['target']['user_profile'][dim])) for t in truth]
            classes = sorted(EMOTIONS) if dim == 'emotion_label' else sorted(set((labels[i] for i in fit_idx)))
            model = NBModel(classes, vocab).fit([features[i] for i in fit_idx], [labels[i] for i in fit_idx])
            best = (-1, None)
            for alpha in [0.1, 0.5, 2.0]:
                model.set_alpha(alpha)
                pred = [model.predict(features[i]) for i in tune_idx]
                acc = sum((p == labels[i] for p, i in zip(pred, tune_idx))) / len(tune_idx)
                tuning.append({'dimension': dim, 'alpha': alpha, 'holdout_accuracy': acc})
                print(tuning[-1], flush=True)
                if acc > best[0]:
                    best = (acc, alpha)
            full_vocab = vocabulary(features, a.vocab_size)
            full_classes = sorted(EMOTIONS) if dim == 'emotion_label' else sorted(set(labels))
            model = NBModel(full_classes, full_vocab, alpha=best[1]).fit(features, labels)
            model.save(model_dir, dim)
            models[dim] = model
        del features
    print('BUILD TRAIN-ONLY RETRIEVAL', flush=True)
    retriever = Retrieval(train, truth)
    preds = []
    timings = []
    for i, s in enumerate(val):
        t0 = time.perf_counter()
        last = grams(text_for(s, 'latest'))
        user = grams(text_for(s, 'users'))
        profile = {d: models[d].predict(user).split('|') for d in DIMS}
        profile = {d: [v for v in vals if v] for d, vals in profile.items()}
        pred = {'id': s['id'], 'response_text': retriever.predict(s), 'emotion_label': models['emotion_label'].predict(last), 'user_profile': profile, 'memory_refs': []}
        json.dumps(pred, ensure_ascii=False)
        timings.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
        if (i + 1) % 500 == 0:
            print(f'PREDICTED {i + 1}/{len(val)}', flush=True)
    write_rows(a.output_dir / 'submission.jsonl', preds)
    from evaluate import evaluate
    report = evaluate(val_gt, preds)
    (a.output_dir / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    majority = Counter((t['target']['emotion_label'] for t in truth)).most_common(1)[0][0]
    naive = [{**p, 'emotion_label': majority, 'user_profile': {d: [] for d in DIMS}} for p in preds]
    naive_report = evaluate(val_gt, naive)
    (a.output_dir / 'majority_empty_metrics.json').write_text(json.dumps(naive_report, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__, 'train_samples': len(train), 'validation_samples': len(val), 'fit_samples': len(fit_idx), 'tune_samples': len(tune_idx), 'group_overlap': 0, 'tuning': tuning, 'selection_policy': '10% SHA256 conversation-group holdout from training; public validation not used for hyperparameter selection.', 'total_seconds': time.perf_counter() - started, 'first100_latency_ms': {'mean': statistics.fmean(timings[:100]), 'median': statistics.median(timings[:100]), 'p95': sorted(timings[:100])[94]}, 'all_samples_average_ms': statistics.fmean(timings), 'official_performance': False, 'data_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.data_dir / 'train_test.jsonl', a.data_dir / 'train_gt.jsonl', a.data_dir / 'val_test.jsonl', a.data_dir / 'val_gt.jsonl']}}
    (a.output_dir / 'experiment.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['samples', 'emotion_accuracy', 'emotion_macro_f1', 'profile_dimension_mean_exact_accuracy', 'bleu4', 'rouge_l_f1']}, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
