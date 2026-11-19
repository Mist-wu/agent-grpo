# agent-grpo：手写多轮工具调用 GRPO

在 Qwen2.5-1.5B-Instruct 上训练会调用工具的 agent：模型可以多轮调用计算器和检索工具，再给出答案。训练分两步：先用答对的轨迹做冷启动 SFT，再用 GRPO 强化学习。GRPO 的采样、奖励、优势和 loss 都在本仓库里手写，没有用 TRL 等现成框架，`src/` 约 1,400 行 Python。

## 当前进度

| 阶段 | 状态 |
|---|---|
| 原模型基线（直接答 / 带工具，各 1,000 题） | 完成 |
| 冷启动 SFT（种子 42） | 完成 |
| GRPO（种子 42，300 步） | 训练中，最新检查点在第 200 步 |
| 种子 43、44 和训练后的完整评测 | 待做 |

训练后的对比结果出来后补到这里。

## 任务和工具

- **GSM8K**：数学应用题，配计算器。计算器用受限的 AST 求值，只支持算术运算。
- **HotpotQA**：多跳问答，配检索。每道题只能检索它自己的候选文档，用 BM25 排序。

一条轨迹最多调用 4 次工具。工具结果插回上下文，模型继续生成，直到给出最终答案。

## GRPO 的实现

- **采样**：每步取 8 道题，每题采样 8 条轨迹，温度 1，不做 top-k / top-p 截断。
- **奖励**：答对得 1 分（HotpotQA 用答案 F1）；轨迹里有能解析的工具调用加 0.1，每条轨迹只加一次；每调用一次工具扣 0.05。扣分是为了压住无意义的重复调用。
- **优势**：同一道题的 8 条轨迹组内标准化。
- **目标函数**：冻结旧策略的 log prob，做 clip（ε = 0.2），加 k3 形式的 KL（β = 0.04）。KL 的参考模型是关掉 LoRA 的基座模型；冷启动 SFT 只用来初始化策略。
- **loss 范围**：只对模型自己生成的 token 算 loss。提示词和插入的工具结果都不算。先在每条轨迹内按 token 平均，再在轨迹间平均。
- **训练设置**：LoRA rank 16，学习率 1e-5，每批采样更新 2 次，每 50 步存一次检查点。

检查点里有 LoRA 权重、优化器状态和全部随机数状态，训练中断后可以从检查点原样续跑。SFT 每 25 步存一次，同样可以续跑。

## 原模型基线

同样 500 道 GSM8K 和 500 道 HotpotQA 测试题，Qwen2.5-1.5B-Instruct 不经训练：

| 条件 | GSM8K 准确率 | HotpotQA 答案 F1 |
|---|---:|---:|
| 直接回答 | 63.8% | 0.136 |
| 带工具 | 18.2% | 0.105 |

原模型拿到工具反而答得更差。检查前 128 条带工具的轨迹，有 35 条用满了 4 次调用，这 35 条没有一条答对。典型情况是对没有文档的 GSM8K 题反复检索。这就是要训练的地方：让模型学会什么时候调工具、调完怎么用结果。详见 [基线结果](results/core-scored-v2/baseline-summary.md) 和 [评分规则](docs/scoring.md)。

## 运行

```sh
python -m pip install -e '.[train]'
python scripts/run.py prepare --out data
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl \
  --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl \
  --out results/eval-base-tools.jsonl --condition base-tools
bash scripts/run_seeds.sh configs/a100.json   # 每个种子：收集轨迹、SFT、GRPO、评测
```

从检查点续跑：

```sh
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl \
  --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter \
  --resume results/grpo-42/checkpoint-0200
```

`configs/a100.json` 是 1.5B 的正式配置，`configs/l4-full.json` 是 0.5B 的小配置。

## 目录

| 路径 | 内容 |
|---|---|
| `src/agent_grpo/objective.py` | 组内优势和 GRPO loss |
| `src/agent_grpo/rollout.py` | 多轮采样和工具调用 |
| `src/agent_grpo/rewards.py` | 奖励函数 |
| `src/agent_grpo/tools.py` | 计算器和检索 |
| `src/agent_grpo/training.py` | 模型加载、检查点 |
| `scripts/` | 运行入口 |
| `results/` | 基线、训练记录和检查点元数据 |
| `docs/` | 数据来源和评分规则 |
