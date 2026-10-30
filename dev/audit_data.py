import json, hashlib, collections, statistics
from pathlib import Path
P = Path(__file__).resolve().parents[1]

def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def main():
    data = P / 'dev/data'
    report = {'splits': {}, 'checks': {}}
    all_hash = {}
    last_hash = {}
    for split in ['train', 'val']:
        samples = rows(data / f'{split}_test.jsonl')
        gt = rows(data / f'{split}_gt.jsonl')
        labels = {r['id']: r['target'] for r in gt}
        assert len({s['id'] for s in samples}) == len(samples)
        assert set(labels) == {s['id'] for s in samples}
        for s in samples:
            assert s['history'][-1]['role'] == 'user'
            assert s['history'][-1]['turn_id'] == s['target_user_turn_id']
            assert all((t['turn_id'] <= s['target_user_turn_id'] for t in s['history']))
        all_hash[split] = collections.defaultdict(list)
        last_hash[split] = collections.defaultdict(list)
        for s in samples:
            content = [(t['role'], t['content']) for t in s['history']]
            all_hash[split][digest(content)].append(s['id'])
            last_hash[split][digest(s['history'][-1]['content'])].append(s['id'])
        counts = collections.Counter((t['emotion_label'] for t in labels.values()))
        report['splits'][split] = {'samples': len(samples), 'conversations': len({s['conversation_id'] for s in samples}), 'emotion_counts': dict(counts), 'input_hashes_duplicated': sum((len(v) - 1 for v in all_hash[split].values())), 'majority_emotion_fraction': counts.most_common(1)[0][1] / len(samples), 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data / f'{split}_test.jsonl', data / f'{split}_gt.jsonl']}}
    common = set(all_hash['train']) & set(all_hash['val'])
    last_common = set(last_hash['train']) & set(last_hash['val'])
    report['checks'] = {'target_boundaries_valid': True, 'unique_ids': True, 'cross_split_identical_full_inputs': len(common), 'validation_ids_with_identical_train_input': [i for k in common for i in all_hash['val'][k]], 'cross_split_identical_last_user_texts': len(last_common), 'note': 'Identical last user text is not by itself leakage; context can differ. Exact full input overlap is reported separately. No future turns or validation labels enter training.'}
    (P / 'reports').mkdir(parents=True, exist_ok=True)
    (P / 'reports/data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
