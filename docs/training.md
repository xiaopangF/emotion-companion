# 训练与复现

## 环境

历史主训练使用RTX4090 24GB单卡、Python 3.12、PyTorch 2.10.0（CUDA 12.8）、
Transformers 4.57.6、PEFT 0.18.1、Accelerate 1.15.0与bitsandbytes 0.49.2。
这是历史环境记录，不是对任意版本组合的兼容保证。

训练环境需安装与驱动兼容的PyTorch，再检查`dev/requirements-train.txt`。
仓库的CPU测试使用Python 3.11，不启动模型服务。

## QLoRA

准备完`dev/data/`后，从仓库根目录执行：

```bash
python dev/train_sft.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --train-file dev/data/train_sft.jsonl \
  --val-file dev/data/val_sft.jsonl \
  --output-dir runs/sft \
  --epochs 1 --max-seq-len 4096 \
  --per-device-batch 1 --grad-accum 8 \
  --lr 0.0001 --warmup-ratio 0.03 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --dtype bf16 --save-steps 100 --eval-steps 100 \
  --max-eval-samples 64
```

使用NF4量化、BF16计算、梯度检查点，仅对completion计算损失，提示词和padding被遮罩。
超出4096 token的训练样本被排除，ID写入`training_meta.json`；并非静默截断。
历史训练保留17960/18657条样本，完成2245步；最佳loss checkpoint为2100。
64条loss只能帮助选择检查点，部署选择仍需生成评估。

产物包括`adapter/`、`checkpoint-*/`、`trainer_state.json`、`training_log.jsonl`、
`step_timing.jsonl`和`training_meta.json`。这些文件包含运行信息，默认不提交Git。

## 两种继续训练方式

| 目的 | 参数 | 优化器与调度器 |
| --- | --- | --- |
| 恢复中断的同一次训练 | `--resume-from-checkpoint runs/sft/checkpoint-100` | 恢复已有状态 |
| 从adapter开始新阶段 | `--init-adapter runs/sft/adapter` | 重新建立 |

两者互斥。新的阶段必须使用独立输出目录：

```bash
python dev/train_sft.py --model /path/to/Qwen2.5-7B-Instruct \
  --init-adapter runs/sft/adapter --output-dir runs/continuation \
  --max-steps 400 --max-seq-len 4096 --grad-accum 8 \
  --lr 0.00002 --dtype bf16 --save-steps 100 --eval-steps 100 \
  --max-eval-samples 64 --max-wall-seconds 3600
```

`--max-wall-seconds`从训练循环开始计时，到更新步边界保存；不包含模型加载、数据准备、
合并与评估，也不会关闭GPU实例。需要总时间限制时，应由任务管理器单独约束完整流程。
上述400步为实验复现示例，历史实验未达到部署条件。

## 合并

```bash
python dev/merge_lora.py --base /path/to/Qwen2.5-7B-Instruct \
  --adapter runs/sft/adapter --output participant/model --dtype bf16
```

使用与训练一致的基座。合并需足够的内存与磁盘空间，输出包含完整权重及tokenizer。
使用新的输出目录，不覆盖唯一的可用模型。

## 画像辅助分类器

```bash
python dev/train_linear.py --data-dir dev/data \
  --output-dir runs/linear_baseline --epochs 5 --threads 4
```

该脚本训练情绪及三个画像维度的字符TF-IDF分类器，epoch通过训练内部分组留出选择。
最终部署只使用`style`，未将线性情绪分类器替代主模型。
要启用style，将以下文件复制到`participant/auxiliary_models/`：

- `runs/linear_baseline/models/style.json`
- `runs/linear_baseline/models/style.npz`
- `runs/linear_baseline/models/users_features.json`

然后将`participant/configs/style.example.json`复制为`inference_config.json`，填写实际硬件信息。
读取特征和权重均为本地操作，不需要访问训练数据。

`dev/baseline.py`保留了字符朴素贝叶斯与训练集检索基线，用于对照；它不是最终生成模型。

## 效果评估

```bash
python dev/evaluate.py outputs/demo/submission.jsonl \
  --gt dev/data/val_gt.jsonl \
  --performance-report outputs/demo/performance_report.json \
  --output outputs/demo/metrics.json
```

输入和目标ID集合必须完全一致。对部分子集生成时，目标也应筛成同一子集。
BERTScore需显式提供本地编码器目录与层数；未提供时结果为`null`，不会假称完成。
本地BLEU/ROUGE定义和画像集合匹配并非经官方确认的综合计分公式。
