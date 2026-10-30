"""Completion-only LoRA/QLoRA SFT with reproducible metadata and explicit dropout audit."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def balance_dataset(rows, alpha, seed):
    """Class mass proportional to count**alpha: alpha=1 natural, 0 uniform."""
    if not 0 <= alpha <= 1:
        raise ValueError('balance-alpha must lie in [0,1]')
    if alpha == 1:
        return list(rows)
    pools = {}
    for row in rows:
        pools.setdefault(row['emotion'], []).append(row)
    rng = random.Random(seed)
    classes = sorted(pools)
    masses = [len(pools[c]) ** alpha for c in classes]
    return [rng.choice(pools[c]) for c in rng.choices(classes, weights=masses, k=len(rows))]

class CompletionCollator:
    """Preserve -100 prompt masks; mask padding by position, never by token value."""

    def __init__(self, pad_token_id, pad_to_multiple_of=8):
        self.pad_id = pad_token_id
        self.multiple = pad_to_multiple_of

    def __call__(self, features):
        import torch
        n = max((len(x['input_ids']) for x in features))
        n = (n + self.multiple - 1) // self.multiple * self.multiple
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for f in features:
            pad = n - len(f['input_ids'])
            batch['input_ids'].append(f['input_ids'] + [self.pad_id] * pad)
            batch['attention_mask'].append(f['attention_mask'] + [0] * pad)
            batch['labels'].append(f['labels'] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

def make_preprocess(tokenizer, max_seq_len):

    def preprocess(example):
        messages = example['messages']
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        if full[:len(prompt)] != prompt:
            raise ValueError('Chat-template prefix mismatch; cannot safely mask prompt')
        if len(full) > max_seq_len:
            return None
        if len(full) <= len(prompt):
            raise ValueError('No completion tokens')
        return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': [-100] * len(prompt) + full[len(prompt):]}
    return preprocess

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--train-file', type=Path, default=ROOT / 'dev/data/train_sft.jsonl')
    p.add_argument('--val-file', type=Path, default=ROOT / 'dev/data/val_sft.jsonl')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--max-seq-len', type=int, default=4096)
    p.add_argument('--per-device-batch', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--lora-targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--balance-alpha', type=float, default=1.0, help='1=natural; 0=uniform; .5=square root. Acc is the primary metric.')
    p.add_argument('--no-qlora', action='store_true')
    p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--save-steps', type=int, default=200)
    p.add_argument('--eval-steps', type=int, default=200)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--max-eval-samples', type=int, default=512)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--init-adapter', type=Path, help='Start a fresh optimizer phase from an existing trainable adapter.')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--max-wall-seconds', type=float, default=0.0, help='Gracefully stop and save at a step boundary after this training wall time; 0 disables.')
    arguments = p.parse_args()
    if arguments.init_adapter and arguments.resume_from_checkpoint:
        p.error('--init-adapter and --resume-from-checkpoint are mutually exclusive')
    return arguments

def main():
    a = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available() and (not a.allow_cpu):
        raise RuntimeError('CUDA unavailable. Install matching CUDA PyTorch on platform; CPU smoke requires --allow-cpu --no-qlora --dtype fp32.')
    if not a.no_qlora and (not torch.cuda.is_available()):
        raise RuntimeError('QLoRA configuration requires CUDA')
    if a.save_steps % a.eval_steps:
        raise ValueError('save-steps must be a multiple of eval-steps')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / 'training_meta.json').exists() and (not a.resume_from_checkpoint):
        raise ValueError('Run already exists: choose a new output directory or resume explicitly')
    set_seed(a.seed)
    dtype = a.dtype
    if dtype == 'auto':
        dtype = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16' if torch.cuda.is_available() else 'fp32'
    td = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    train = read_rows(a.train_file)
    val = read_rows(a.val_file)
    raw_n = len(train)
    if a.max_samples:
        train = random.Random(a.seed).sample(train, min(a.max_samples, len(train)))
    train = balance_dataset(train, a.balance_alpha, a.seed)
    if a.max_eval_samples and len(val) > a.max_eval_samples:
        val = random.Random(a.seed).sample(val, a.max_eval_samples)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    prep = make_preprocess(tok, a.max_seq_len)

    def convert(rows):
        kept = []
        dropped = []
        for r in rows:
            x = prep(r)
            if x is None:
                dropped.append(r['sample_id'])
            else:
                kept.append(x)
        if not kept:
            raise ValueError('No examples fit max-seq-len')
        return (kept, dropped)
    train_ds, drop_train = convert(train)
    val_ds, drop_val = convert(val)
    kws = {'trust_remote_code': a.trust_remote_code, 'torch_dtype': td, 'attn_implementation': 'sdpa'}
    if not a.no_qlora:
        kws['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=td)
        kws['device_map'] = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kws)
    if not a.no_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    if a.init_adapter:
        if not (a.init_adapter / 'adapter_config.json').is_file() or not (a.init_adapter / 'adapter_model.safetensors').is_file():
            raise ValueError('Initial adapter requires configuration and safetensors weights')
        model = PeftModel.from_pretrained(model, str(a.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=a.lora_targets.split(','), bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    kwargs = dict(output_dir=str(a.output_dir), num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.per_device_batch, per_device_eval_batch_size=a.per_device_batch, gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, warmup_ratio=a.warmup_ratio, lr_scheduler_type='cosine', logging_steps=a.logging_steps, save_strategy='steps', save_steps=a.save_steps, eval_steps=a.eval_steps, bf16=dtype == 'bf16', fp16=dtype == 'fp16', gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, max_grad_norm=1.0, seed=a.seed, data_seed=a.seed, report_to=[], prediction_loss_only=True, remove_unused_columns=False, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, dataloader_num_workers=0)
    key = 'eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments).parameters else 'evaluation_strategy'
    kwargs[key] = 'steps'
    ta = TrainingArguments(**kwargs)

    class LogCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.training_started = time.perf_counter()
            self.step_started = self.training_started
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, args, state, control, **kw):
            self.step_started = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            elapsed = time.perf_counter() - self.training_started
            record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                      'step': state.global_step,
                      'optimizer_step_seconds': time.perf_counter() - self.step_started,
                      'training_elapsed_seconds': elapsed,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
            with (a.output_dir / 'step_timing.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            if a.max_wall_seconds > 0 and elapsed >= a.max_wall_seconds:
                control.should_save = True
                control.should_evaluate = True
                control.should_training_stop = True
                record['stop_reason'] = 'Configured training wall time reached at optimizer step boundary'
                (a.output_dir / 'budget_stop.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                with (a.output_dir / 'training_log.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'step': state.global_step, **logs}) + '\n')
    meta = {'created_utc': datetime.now(timezone.utc).isoformat(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, 'train_raw': raw_n, 'train_used': len(train_ds), 'training_sample_selection': 'seeded random when max-samples is set; full public training otherwise', 'validation_used_for_loss': len(val_ds), 'train_dropped_ids': drop_train, 'validation_dropped_ids': drop_val, 'dtype_resolved': dtype, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.version, 'data_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.train_file, a.val_file]}, 'selection_note': 'Best eval_loss checkpoint only; choose deployment through full generation metrics, not loss alone.'}
    if a.init_adapter:
        meta['initial_adapter_sha256'] = hashlib.sha256((a.init_adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
        meta['optimizer_phase'] = 'Fresh optimizer and scheduler; initialized trainable adapter, not resumed optimizer state.'
    (a.output_dir / 'training_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    tk = dict(model=model, args=ta, train_dataset=train_ds, eval_dataset=val_ds, data_collator=CompletionCollator(tok.pad_token_id), callbacks=[LogCallback()])
    if 'processing_class' in inspect.signature(Trainer).parameters:
        tk['processing_class'] = tok
    else:
        tk['tokenizer'] = tok
    trainer = Trainer(**tk)
    result = trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_metrics('train', result.metrics)
    trainer.save_model(str(a.output_dir / 'adapter'))
    tok.save_pretrained(a.output_dir / 'adapter')
    print(f"SAVED {a.output_dir / 'adapter'}")
if __name__ == '__main__':
    main()
