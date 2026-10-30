#!/usr/bin/env python3
"""把 train/val 公开数据解析成 SFT 指令样本，并输出统计信息。

纯标准库实现，不依赖 torch/transformers，便于在任意机器上先跑通数据。
输出:
  dev/data/train_sft.jsonl   每行 {"messages":[{system},{user},{assistant}]}
  dev/data/val_sft.jsonl     同上
  dev/data/val_gt.jsonl      每行 {id, target}，用于本地评测
  dev/data/{train,val}_stats.json  标签分布 / 长度统计
"""
from __future__ import annotations

import json
import argparse
import sys
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "训练-验证-数据集" / "train" / "train_public.jsonl"
VAL = ROOT / "训练-验证-数据集" / "val" / "val_public.jsonl"
OUT = ROOT / "dev" / "data"

sys.path.insert(0, str(ROOT / "participant"))
from run_inference import EMOTIONS, PROFILE_LABELS, SYSTEM_PROMPT, conversation_text as inference_text


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        emotions=", ".join(sorted(EMOTIONS)),
        personality=", ".join(sorted(PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(PROFILE_LABELS["style"])),
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 解析失败: {exc}") from exc
            rows.append(row)
    return rows


def history_until(turns: list[dict], target_user_turn_id: int) -> list[dict]:
    return [t for t in turns if t["turn_id"] <= target_user_turn_id]


def conversation_text(sample_id: str, turns: list[dict]) -> str:
    return inference_text({"id": sample_id, "history": turns})


def target_object(target: dict) -> dict:
    profile = target.get("user_profile") or {}
    return {
        "emotion_label": target.get("emotion_label", ""),
        "user_profile": {
            "personality_traits": profile.get("personality_traits") or [],
            "interests": profile.get("interests") or [],
            "style": profile.get("style") or [],
        },
        "response_text": target.get("response_text", ""),
        # 测试集没有 memory bank，统一教模型输出空数组，与测试分布对齐。
        "memory_refs": [],
    }


def char_count(text: str) -> int:
    return len(text)


def build_dataset(path: Path, split: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    conversations = read_jsonl(path)
    samples: list[dict] = []
    ground_truth: list[dict] = []
    test_rows: list[dict] = []
    emotion_counter: Counter = Counter()
    profile_counter = {k: Counter() for k in PROFILE_LABELS}
    response_lens: list[int] = []
    history_turn_counts: list[int] = []
    history_char_counts: list[int] = []
    memory_ref_counts: list[int] = []
    empty_profile_dims = Counter()

    sys_prompt = system_prompt()
    for conv in conversations:
        turns = conv["turns"]
        for point in conv["prediction_points"]:
            tid = point["target_user_turn_id"]
            hist = history_until(turns, tid)
            target = point["target"]
            sample_id = f"{conv['conversation_id']}_t{tid}"
            if not hist or hist[-1]["role"] != "user" or hist[-1]["turn_id"] != tid:
                raise ValueError(f"Invalid target boundary: {sample_id}")
            if any(t["turn_id"] > tid for t in hist):
                raise ValueError(f"Future turn leakage: {sample_id}")

            user_text = conversation_text(sample_id, hist)
            assistant_obj = target_object(target)
            emotion = target.get("emotion_label", "")
            samples.append({
                "sample_id": sample_id,
                "conversation_id": conv["conversation_id"],
                "emotion": emotion,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
                ]
            })
            test_rows.append({
                "id": sample_id,
                "conversation_id": conv["conversation_id"],
                "target_user_turn_id": tid,
                "history": [
                    {"turn_id": t["turn_id"], "role": t["role"], "content": t["content"]}
                    for t in hist
                ],
            })
            ground_truth.append({"id": sample_id, "target": target_object(target)})

            emotion_counter[target.get("emotion_label", "")] += 1
            profile = target.get("user_profile") or {}
            for dim in PROFILE_LABELS:
                labels = profile.get(dim) or []
                if not labels:
                    empty_profile_dims[dim] += 1
                for label in labels:
                    profile_counter[dim][label] += 1
            response_lens.append(char_count(target.get("response_text", "")))
            history_turn_counts.append(len(hist))
            history_char_counts.append(sum(char_count(t["content"]) for t in hist))
            memory_ref_counts.append(len(target.get("memory_refs") or []))

    def mean_median_p95(values: list[int]) -> dict:
        if not values:
            return {"mean": None, "median": None, "p95": None}
        ordered = sorted(values)
        p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
        return {
            "mean": round(statistics.mean(values), 2),
            "median": statistics.median(values),
            "p95": p95,
        }

    stats = {
        "split": split,
        "conversations": len(conversations),
        "samples": len(samples),
        "emotion_distribution": dict(emotion_counter.most_common()),
        "profile_distribution": {
            dim: dict(counter.most_common()) for dim, counter in profile_counter.items()
        },
        "empty_profile_dims": dict(empty_profile_dims),
        "response_char_len": mean_median_p95(response_lens),
        "history_turn_count": mean_median_p95(history_turn_counts),
        "history_char_count": mean_median_p95(history_char_counts),
        "memory_ref_counts": dict(Counter(memory_ref_counts).most_common()),
    }
    return samples, ground_truth, test_rows, stats


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "训练-验证-数据集")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        path = args.data_root / split / f"{split}_public.jsonl"
        samples, ground_truth, test_rows, stats = build_dataset(path, split)
        dump_jsonl(OUT / f"{split}_sft.jsonl", samples)
        dump_jsonl(OUT / f"{split}_gt.jsonl", ground_truth)
        dump_jsonl(OUT / f"{split}_test.jsonl", test_rows)
        (OUT / f"{split}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"== {split} ==")
        print(f"  conversations={stats['conversations']} samples={stats['samples']}")
        print(f"  emotions={dict(stats['emotion_distribution'])}")
        print(f"  response_len={stats['response_char_len']}")
        print(f"  history_turns={stats['history_turn_count']}")
        print(f"  history_chars={stats['history_char_count']}")
        print(f"  empty_profile_dims={dict(stats['empty_profile_dims'])}")
        print(f"  memory_ref_counts={dict(stats['memory_ref_counts'])}")
        print()


if __name__ == "__main__":
    main()
