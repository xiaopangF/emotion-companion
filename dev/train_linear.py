"""Train sparse TF-IDF softmax models on CPU. Select epochs on grouped training holdout."""
import argparse, hashlib, json, math, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import rows, write_rows, grams, text_for, Retrieval, DIMS
from data_split import holdout_indices
from evaluate import evaluate
P = Path(__file__).resolve().parents[1]

class SparseClassifier(torch.nn.Module):

    def __init__(self, n_vocab, n_classes):
        super().__init__()
        self.weight = torch.nn.EmbeddingBag(n_vocab, n_classes, mode='sum', sparse=True)
        torch.nn.init.zeros_(self.weight.weight)
        self.bias = torch.nn.Parameter(torch.zeros(n_classes))

    def forward(self, indices, offsets, values):
        return self.weight(indices, offsets, per_sample_weights=values) + self.bias

def make_vocab(samples, indices, mode, limit):
    df = Counter()
    for i in indices:
        df.update(grams(text_for(samples[i], mode)).keys())
    terms = [g for g, n in df.most_common(limit) if n >= 2]
    vocab = {g: i for i, g in enumerate(terms)}
    idf = np.asarray([math.log((1 + len(indices)) / (1 + df[g])) + 1 for g in terms], dtype=np.float32)
    return (vocab, idf)

def encode(sample, mode, vocab, idf):
    counts = grams(text_for(sample, mode))
    pairs = [(vocab[g], (1 + math.log(n)) * idf[vocab[g]]) for g, n in counts.items() if g in vocab]
    if not pairs:
        return (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32))
    ids, values = zip(*pairs)
    v = np.asarray(values, dtype=np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-08)
    return (np.asarray(ids, dtype=np.int64), v)

def batch(features, indices):
    idx = np.concatenate([features[i][0] for i in indices])
    val = np.concatenate([features[i][1] for i in indices])
    offsets = np.cumsum([0] + [len(features[i][0]) for i in indices[:-1]], dtype=np.int64)
    return (torch.from_numpy(idx), torch.from_numpy(offsets), torch.from_numpy(val))

@torch.no_grad()
def predict(model, features, indices, bs=256):
    model.eval()
    preds = []
    for j in range(0, len(indices), bs):
        preds.extend(model(*batch(features, indices[j:j + bs])).argmax(-1).tolist())
    return preds

def train_epochs(features, y, fit, tune, classes, vocab_n, epochs, seed, log_path, dimension, phase, bs=256):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SparseClassifier(vocab_n, len(classes))
    opt = torch.optim.SparseAdam(model.weight.parameters(), lr=0.025)
    bias_opt = torch.optim.Adam([model.bias], lr=0.025)
    best_acc = -1
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(fit)
        rng.shuffle(order)
        losses = []
        for j in range(0, len(order), bs):
            selection = order[j:j + bs]
            opt.zero_grad()
            bias_opt.zero_grad()
            logits = model(*batch(features, selection))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([y[i] for i in selection]))
            loss.backward()
            opt.step()
            bias_opt.step()
            losses.append(float(loss.detach()))
        acc = None
        if tune:
            preds = predict(model, features, tune)
            acc = sum((p == y[i] for p, i in zip(preds, tune))) / len(tune)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
        item = {'utc': datetime.now(timezone.utc).isoformat(), 'dimension': dimension, 'phase': phase, 'epoch': epoch, 'mean_loss': float(np.mean(losses)), 'training_holdout_accuracy': acc}
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item) + '\n')
        print(item, flush=True)
    return (model, best_epoch, best_acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=P / 'dev/data')
    ap.add_argument('--output-dir', type=Path, default=P / 'runs/linear_baseline')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--vocab-size', type=int, default=40000)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    log = a.output_dir / 'training_log.jsonl'
    if log.exists():
        raise ValueError('Use a fresh run directory')
    t0 = time.perf_counter()
    samples = rows(a.data_dir / 'train_test.jsonl')
    gt = rows(a.data_dir / 'train_gt.jsonl')
    gtmap = {r['id']: r['target'] for r in gt}
    targets = [gtmap[s['id']] for s in samples]
    val = rows(a.data_dir / 'val_test.jsonl')
    vgt = rows(a.data_dir / 'val_gt.jsonl')
    fit, tune, keys = holdout_indices(samples)
    assert {keys[i] for i in fit}.isdisjoint({keys[i] for i in tune})
    models = {}
    selected = {}
    predicted = {}
    model_dir = a.output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    for mode, dims in [('latest', ['emotion_label']), ('users', list(DIMS))]:
        vocab, idf = make_vocab(samples, fit, mode, a.vocab_size)
        features = [encode(s, mode, vocab, idf) for s in samples]
        vfeatures = [encode(s, mode, vocab, idf) for s in val]
        (model_dir / f'{mode}_features.json').write_text(json.dumps({'vocab': vocab, 'idf': idf.tolist(), 'mode': mode}, ensure_ascii=False), encoding='utf-8')
        for d in dims:
            labels = [t['emotion_label'] if d == 'emotion_label' else '|'.join(sorted(t['user_profile'][d])) for t in targets]
            classes = sorted(set((labels[i] for i in fit)))
            idx = {c: i for i, c in enumerate(classes)}
            y = [idx.get(l, -1) for l in labels]
            _, best_epoch, acc = train_epochs(features, y, fit, tune, classes, len(vocab), a.epochs, 20261014, log, d, 'tune')
            classes = sorted(set(labels))
            idx = {c: i for i, c in enumerate(classes)}
            y = [idx[l] for l in labels]
            model, _, _ = train_epochs(features, y, list(range(len(samples))), [], classes, len(vocab), best_epoch, 20261014, log, d, 'refit')
            pred = predict(model, vfeatures, list(range(len(val))))
            predicted[d] = [classes[i] for i in pred]
            selected[d] = {'epochs': best_epoch, 'holdout_accuracy': acc}
            np.savez_compressed(model_dir / f'{d}.npz', weight=model.weight.weight.detach().numpy(), bias=model.bias.detach().numpy())
            (model_dir / f'{d}.json').write_text(json.dumps({'classes': classes, 'feature_mode': mode}, ensure_ascii=False), encoding='utf-8')
        del features, vfeatures
    previous = P / 'runs/char_baseline/submission.jsonl'
    if previous.exists():
        responses = {r['id']: r['response_text'] for r in rows(previous)}
    else:
        retriever = Retrieval(samples, gt)
        responses = {s['id']: retriever.predict(s) for s in val}
    preds = [{'id': s['id'], 'response_text': responses[s['id']], 'emotion_label': predicted['emotion_label'][i], 'user_profile': {d: [v for v in predicted[d][i].split('|') if v] for d in DIMS}, 'memory_refs': []} for i, s in enumerate(val)]
    write_rows(a.output_dir / 'submission.jsonl', preds)
    result = evaluate(vgt, preds)
    (a.output_dir / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    meta = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'fit_examples': len(fit), 'tune_examples': len(tune), 'total_train_examples': len(samples), 'selected': selected, 'total_seconds': time.perf_counter() - t0, 'torch': torch.__version__, 'device': 'CPU', 'hyperparameter_selection': 'Training holdout grouped by conversation AND identical history; public validation only evaluated after config selection.', 'response_source': 'Same train-only retrieval as char_baseline; classifier experiment only.', 'data_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.data_dir.glob('*_gt.jsonl')}}
    (a.output_dir / 'experiment.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: result[k] for k in ['emotion_accuracy', 'profile_dimension_mean_exact_accuracy']}, flush=True)
if __name__ == '__main__':
    main()
