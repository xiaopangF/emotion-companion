#!/usr/bin/env python3
"""把 LoRA adapter 合并回基座模型，导出为可部署的完整模型目录。

用法:
  python dev/merge_lora.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --adapter runs/qwen2.5-7b-sft/adapter \
    --output participant/model \
    --dtype bf16

合并后即可用 start.sh 的 vLLM 后端直接服务 participant/model。
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 LoRA 到基座模型")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"加载基座: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=False
    )
    print(f"加载 adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    tokenizer.save_pretrained(args.output)
    print(f"合并完成 -> {args.output}")


if __name__ == "__main__":
    main()
