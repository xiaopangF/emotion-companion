"""Versioned sparse features shared by CPU training and deployed classification."""
import math
from collections import Counter
import numpy as np


def char_ngrams(text):
    text = ''.join(text.lower().split())
    return Counter(text[i:i+n] for n in (1, 2, 3) for i in range(len(text)-n+1))


def feature_blocks(history, mode):
    users = [turn['content'] for turn in history if turn['role'] == 'user']
    latest = users[-1] if users else ''
    if mode == 'latest':
        return [('', 1.0, latest)]
    if mode == 'users':
        return [('', 1.0, '\n'.join(users)[-6000:])]
    if mode == 'emotion_context_v2':
        assistants = [turn['content'] for turn in history[:-1] if turn['role'] == 'assistant']
        return [('L:', 1.0, latest), ('P:', 0.35, '\n'.join(users[-3:-1])[-1500:]),
                ('A:', 0.25, assistants[-1][-1500:] if assistants else '')]
    if mode == 'profile_context_v2':
        return [('H:', 1.0, '\n'.join(users)[-6000:]), ('L:', 0.65, latest),
                ('R:', 0.35, '\n'.join(users[-3:])[-2400:])]
    raise ValueError(f'Unknown feature mode: {mode}')


def feature_keys(history, mode):
    return {prefix + gram for prefix, _, text in feature_blocks(history, mode)
            for gram in char_ngrams(text)}


def encode_history(history, encoder):
    vocab = encoder['vocab']
    idf = encoder['idf']
    ids_all, values_all = [], []
    for prefix, scale, text in feature_blocks(history, encoder['mode']):
        pairs = [(vocab[prefix+gram], (1+math.log(n))*idf[vocab[prefix+gram]])
                 for gram, n in char_ngrams(text).items() if prefix+gram in vocab]
        if not pairs:
            continue
        indices, values = zip(*pairs)
        values = np.asarray(values, dtype=np.float32)
        values *= scale / max(float(np.linalg.norm(values)), 1e-8)
        ids_all.extend(indices)
        values_all.extend(values)
    return np.asarray(ids_all, dtype=np.int64), np.asarray(values_all, dtype=np.float32)


def decode_scores(logits, metadata):
    classes = metadata['classes']
    if metadata.get('head_type', 'softmax') == 'multilabel':
        threshold = float(metadata.get('threshold', 0.5))
        if not 0 < threshold < 1:
            raise ValueError('threshold must lie in (0, 1)')
        cutoff = math.log(threshold / (1-threshold))
        return [label for label, score in zip(classes, logits) if score >= cutoff]
    label = classes[int(np.argmax(logits))]
    return label if metadata['field'] == 'emotion_label' else [x for x in label.split('|') if x]
