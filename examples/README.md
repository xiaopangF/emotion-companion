# 虚构格式样例

本目录全部内容是为说明接口人工编写的示例，不来自训练集、验证集或真实用户。
`predictions.jsonl`也是手写样例，并非某个模型的输出。
`targets.jsonl`用于演示评估器，示例指标不代表项目准确率。

```bash
python dev/evaluate.py examples/predictions.jsonl --gt examples/targets.jsonl
```
