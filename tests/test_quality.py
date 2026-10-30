import importlib.util, json, math, random, sys, unittest
from pathlib import Path
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'dev'))
sys.path.insert(0, str(P / 'participant'))
from evaluate import corpus_bleu, lcs_len, evaluate, validate_row, index_unique
from train_sft import CompletionCollator, balance_dataset, make_preprocess
from prepare_data import system_prompt, target_object, conversation_text
from run_inference import SYSTEM_PROMPT, EMOTIONS, PROFILE_LABELS, parse_prediction, api_generation_options

class QualityTests(unittest.TestCase):

    def test_bleu_identity(self):
        self.assertAlmostEqual(corpus_bleu([list('abcdabcd')], [list('abcdabcd')])['score'], 100.0)

    def test_bleu_short_penalty(self):
        r = corpus_bleu([list('abcdefgh')], [list('abcd')])
        self.assertAlmostEqual(r['brevity_penalty'], math.exp(-1))
        self.assertAlmostEqual(r['score'], 100 * math.exp(-1))

    def test_bleu_zero_overlap(self):
        self.assertEqual(corpus_bleu([list('abcd')], [list('efgh')])['score'], 0)

    def test_lcs_matches_dynamic_programming(self):
        rng = random.Random(42)
        for _ in range(100):
            a = [rng.choice('abc') for i in range(rng.randrange(30))]
            b = [rng.choice('abc') for i in range(rng.randrange(30))]
            prev = [0] * (len(b) + 1)
            for x in a:
                cur = [0]
                for j, y in enumerate(b):
                    cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[-1]))
                prev = cur
            self.assertEqual(lcs_len(a, b), prev[-1])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            index_unique([{'id': 'a'}, {'id': 'a'}])

    def test_balance_endpoints(self):
        rows = [{'emotion': 'a', 'i': i} for i in range(900)] + [{'emotion': 'b', 'i': i} for i in range(100)]
        self.assertEqual(balance_dataset(rows, 1, 42), rows)
        out = balance_dataset(rows, 0, 42)
        self.assertTrue(420 < sum((x['emotion'] == 'a' for x in out)) < 580)
        with self.assertRaises(ValueError):
            balance_dataset(rows, -0.1, 42)

    def test_prompt_is_shared(self):
        expected = SYSTEM_PROMPT.format(emotions=', '.join(sorted(EMOTIONS)), personality=', '.join(sorted(PROFILE_LABELS['personality_traits'])), interests=', '.join(sorted(PROFILE_LABELS['interests'])), styles=', '.join(sorted(PROFILE_LABELS['style'])))
        self.assertEqual(system_prompt(), expected)
        self.assertNotIn('secret_id', conversation_text('secret_id', [{'role': 'user', 'content': 'hello'}]))

    def test_target_order(self):
        t = target_object({'response_text': '你好', 'emotion_label': 'joy'})
        self.assertEqual(list(t), ['emotion_label', 'user_profile', 'response_text', 'memory_refs'])

    def test_prediction_parse(self):
        obj = {'emotion_label': 'anxiety', 'user_profile': {'personality_traits': [], 'interests': [], 'style': []}, 'response_text': '你好', 'memory_refs': []}
        parsed, failed = parse_prediction('test1_000001_t5', '```json\n' + json.dumps(obj) + '\n```')
        self.assertFalse(failed)
        validate_row(parsed)

    def test_bad_json_preserves_reply(self):
        p, f = parse_prediction('x', '{"response_text":"hello')
        self.assertTrue(f)
        self.assertEqual(p['response_text'], 'hello')

    def test_greedy_api(self):
        self.assertEqual(api_generation_options({'do_sample': False, 'max_new_tokens': 512})['temperature'], 0)

    def test_preprocess_boundary(self):

        class Tok:

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 0]
        f = make_preprocess(Tok(), 10)({'messages': [{'role': 'user'}, {'role': 'assistant'}]})
        self.assertEqual(f['labels'], [-100, -100, -100, 4, 0])
        self.assertIsNone(make_preprocess(Tok(), 3)({'messages': [{}, {}]}))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch absent in this interpreter')
    def test_collator_masks_only_padding(self):
        features = [{'input_ids': [7, 8, 9, 0], 'attention_mask': [1] * 4, 'labels': [-100, -100, 9, 0]}, {'input_ids': [3, 0], 'attention_mask': [1, 1], 'labels': [-100, 0]}]
        out = CompletionCollator(0, pad_to_multiple_of=4)(features)
        self.assertEqual(out['labels'].tolist(), [[-100, -100, 9, 0], [-100, 0, -100, -100]])
        self.assertEqual(out['attention_mask'].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_metrics_perfect_fixture(self):
        p = {'id': 'fixture', 'response_text': '今天辛苦了慢慢来', 'emotion_label': 'relaxed', 'user_profile': {k: [] for k in PROFILE_LABELS}, 'memory_refs': []}
        r = evaluate([{'id': 'fixture', 'target': p}], [p])
        self.assertEqual(r['emotion_accuracy'], 1)
        self.assertEqual(r['profile_dimension_mean_exact_accuracy'], 1)
        self.assertEqual(r['rouge_l_f1'], 100)
        self.assertIsNone(r['bertscore'])
if __name__ == '__main__':
    unittest.main(verbosity=2)
