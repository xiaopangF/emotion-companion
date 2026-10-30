# 数据与输出格式

## 推理输入

JSONL每行一个对象。必需字段为唯一字符串`id`和`history`数组。
历史仅包含当前用户轮及之前的消息；最后一轮应为用户消息，不得包含待预测的回复。

```json
{"id":"demo_001","history":[{"turn_id":1,"role":"user","content":"明天要做展示，我有些紧张。"}]}
```

`history`消息使用`role`（`user`或`assistant`）和字符串`content`。
`conversation_id`、`target_user_turn_id`用于准备训练数据和按对话分组，不进入模型提示。

## 推理输出

```json
{
  "id": "demo_001",
  "emotion_label": "anxiety",
  "user_profile": {"personality_traits": [], "interests": [], "style": []},
  "response_text": "第一次展示前紧张很正常，可以先把开场白练习一遍。",
  "memory_refs": []
}
```

情绪枚举和画像标签以`participant/run_inference.py`中的`EMOTIONS`与`PROFILE_LABELS`为准。
画像没有充分证据时允许空数组。没有公开memory bank，因此当前任务不执行记忆检索，
训练目标使用空`memory_refs`。

## 训练原始格式

`dev/prepare_data.py`读取以下目录，数据由使用者自行准备：

```text
dataset/
├── train/train_public.jsonl
└── val/val_public.jsonl
```

每行是一段对话：

```json
{
  "conversation_id": "example_conversation",
  "turns": [{"turn_id": 1, "role": "user", "content": "明天展示有些紧张。"}],
  "prediction_points": [{
    "target_user_turn_id": 1,
    "target": {
      "emotion_label": "anxiety",
      "user_profile": {"personality_traits": [], "interests": [], "style": []},
      "response_text": "可以先练习开场，把最想讲清楚的内容列成三个要点。",
      "memory_refs": []
    }
  }]
}
```

转换产物写入`dev/data/`：`*_sft.jsonl`用于微调，`*_test.jsonl`用于推理，
`*_gt.jsonl`仅用于评估。转换时校验轮次边界，防止未来消息进入输入。

## 划分与审计

`dev/data_split.py`将同一对话及完整输入重复的对话归到同组；
CPU基线在训练内部按组留出调参，公开验证答案不进入优化器。
`python dev/audit_data.py`检查完整输入重复、ID与轮次边界，报告写入`reports/`。

已被主模型训练过的对话，不能通过重新划分变成该模型的独立测试。
训练、验证和真实使用数据均默认被Git忽略；不要强制加入原始对话。
