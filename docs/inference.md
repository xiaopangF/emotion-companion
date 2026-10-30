# 推理与部署

## 默认配置

`participant/configs/inference_config.json`默认使用本地OpenAI兼容接口，
关闭需要额外权重的辅助分类器。启用历史style方案见[训练说明](training.md#画像辅助分类器)。
`api_key`默认空；如服务要求认证，可用`EVALUATION_API_KEY`环境变量提供，不写入Git。

## Linux统一入口

```bash
bash participant/start.sh examples/input.jsonl outputs/demo
```

入口要求Conda环境`conda_env`，并依赖该环境里的vLLM。
脚本自行启动本地模型服务，完成推理后关闭其启动的服务；不会关闭实例。
默认参数为BF16、32768上下文、显存利用比例0.85、最多8条序列及eager模式。
这些是历史Qwen2.5-7B/4090参数，换用其他模型或硬件需重新验证。

脚本启动服务时会读取默认配置，客户端支持`EVALUATION_CONFIG_PATH`覆盖。
若自定义配置同时改了模型路径，请修改默认配置，或显式设置`EVALUATION_MODEL_PATH`，
避免服务和客户端使用不同模型。

## 使用已经启动的服务

Windows或不采用Conda启动脚本时，可在服务启动后直接运行Python客户端：

```bash
EVALUATION_BASE_URL=http://127.0.0.1:8000/v1 \
EVALUATION_API_MODEL=model \
python participant/run_inference.py examples/input.jsonl outputs/demo
```

PowerShell对应写法：

```powershell
$env:EVALUATION_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:EVALUATION_API_MODEL = 'model'
python participant/run_inference.py examples/input.jsonl outputs/demo
```

Python入口本身不会启动或关闭已有服务。Transformers后端也保留在代码中，
可通过配置`backend: transformers`直接加载本地模型；受限JSON重试目前仅适用于兼容API后端。

## 文件与计时约定

处理全部输入，前100条用于计时，计时包含prompt构建、请求、推理、解析、重试与辅助分类。
服务加载和启动耗时不计入。少于100条时`complete=false`。

已有结果目录允许复用：仅覆盖`submission.jsonl`和`performance_report.json`；
其他文件保留。发生异常时创建独立`inference_evidence_*`目录。
该目录可能含原始生成文本，应按对话数据保护，不要提交仓库。

无法解析或字段不合法时，兼容API后端最多尝试一次受限重试；
仍失败时保留可恢复字段，记录ID并继续下一条，不伪造标签。
网络错误、模型缺失等运行错误仍会显式报错。

## 比赛镜像布局

```text
/root/
├── participant/
│   ├── start.sh
│   ├── run_inference.py
│   ├── configs/inference_config.json
│   ├── model/                 # 完整实际模型文件
│   └── auxiliary_models/      # 启用辅助分类时需要
└── AI技术赛道-数字人综合情感对话模型-队名/
```

固定命令：`bash /root/participant/start.sh TEST_FILE RESULT_DIR`。
模型、辅助权重与依赖需位于镜像内，不能链接到训练挂载路径。
专家材料应为正式技术报告、复现代码和真实实验记录，不包含私人操作说明。

```bash
python scripts/check_submission.py --participant participant
```

静态检查仅核对文件，不证明GPU可运行、断网可运行或平台审核通过。
历史v2.1曾在隔离网络、隐藏训练目录的当前容器中完成验收；
本Git仓库不包含Docker镜像，也不冒充已通过官方测评的镜像。

比赛使用平台提供的vLLM基础镜像及一键上传流程，未确认可公开使用的镜像仓库标签，
因此没有编造一个通用Dockerfile来替代比赛基础环境。
