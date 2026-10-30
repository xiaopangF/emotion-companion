"""Group related conversations and exact duplicate input histories before holdout split."""
import hashlib, json

def group_keys(samples):
    parent = {s['conversation_id']: s['conversation_id'] for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = (find(a), find(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen = {}
    for s in samples:
        payload = json.dumps([(t['role'], t['content']) for t in s['history']], ensure_ascii=False, separators=(',', ':'))
        key = hashlib.sha256(payload.encode()).hexdigest()
        cid = s['conversation_id']
        if key in seen:
            union(cid, seen[key])
        else:
            seen[key] = cid
    return [find(s['conversation_id']) for s in samples]

def holdout_indices(samples):
    keys = group_keys(samples)
    tune = [i for i, k in enumerate(keys) if int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 10 == 0]
    tune_set = set(tune)
    fit = [i for i in range(len(samples)) if i not in tune_set]
    return (fit, tune, keys)
