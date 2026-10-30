"""Optional sparse classifiers. Predictions depend only on supplied conversation history."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
from sparse_features import encode_history, decode_scores

def features(history, mode):
    users = [t['content'] for t in history if t['role'] == 'user']
    text = (users[-1] if users else '') if mode == 'latest' else '\n'.join(users)[-6000:]
    text = ''.join(text.lower().split())
    return Counter((text[i:i + n] for n in (1, 2, 3) for i in range(len(text) - n + 1)))

class AuxiliaryClassifiers:

    def __init__(self, path, fields):
        self.path = Path(path)
        self.fields = fields
        self.encoders = {}
        self.models = {}
        for name in fields:
            info = json.loads((self.path / f'{name}.json').read_text(encoding='utf-8'))
            mode = info['feature_mode']
            if mode not in self.encoders:
                enc = json.loads((self.path / f'{mode}_features.json').read_text(encoding='utf-8'))
                enc['idf'] = np.asarray(enc['idf'], dtype=np.float32)
                self.encoders[mode] = enc
            with np.load(self.path / f'{name}.npz', allow_pickle=False) as z:
                self.models[name] = (info, z['weight'], z['bias'])

    def predict(self, history):
        encoded = {}
        result = {}
        for name, (info, weights, bias) in self.models.items():
            mode = info['feature_mode']
            if mode not in encoded:
                encoded[mode] = encode_history(history, self.encoders[mode])
            indices, values = encoded[mode]
            logits = values @ weights[indices] + bias
            result[name] = decode_scores(logits, {**info, 'field': name})
        return result
