# Webshop Environment for RLLM

本模块将 [WebShop](https://github.com/princeton-nlp/webshop) 购物环境集成到 RLLM 多任务训练框架中。

## 概述

WebShop 是一个模拟电商网站环境，包含 118 万个真实产品和 12,087 条众包标注的购物指令。Agent 需要通过搜索、浏览、选择产品选项等操作，找到并购买符合用户需求的产品。

## 安装依赖

### 1. 安装 Webshop 环境依赖

```bash
# 进入 webshop 目录
cd agent_system/environments/env_package/webshop/webshop

# 创建 conda 环境（推荐）
conda create -n webshop python=3.8.13
conda activate webshop

# 运行安装脚本
# -d small: 下载 1000 个产品（用于测试）
# -d all: 下载全部 118 万个产品
./setup.sh -d small
```

### 2. 主要依赖包

| 包名 | 版本 | 用途 |
|------|------|------|
| gym | 0.24.0 | OpenAI Gym 环境接口 |
| Flask | 2.1.2 | Web 服务器 |
| spacy | 3.7.2 | NLP 处理（奖励计算中的文本匹配） |
| thefuzz | 0.19.0 | 模糊字符串匹配（属性匹配） |
| pyserini | 0.17.0 | 搜索引擎 |
| beautifulsoup4 | 4.11.1 | HTML 解析 |
| openjdk | 11 (conda) | Java 运行时（搜索引擎需要） |

### 3. 下载 spaCy 模型

```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```

## 数据集

### 数据来源

Webshop 的数据通过 `setup.sh` 脚本从 Google Drive 下载：

| 文件 | 说明 |
|------|------|
| `items_shuffle.json` | 产品信息（118万个产品） |
| `items_ins_v2.json` | 产品属性 |
| `items_human_ins.json` | 人工标注的购物指令（12,087条） |

### 数据划分

| 划分 | Goal Index 范围 | 数量 |
|------|-----------------|------|
| 测试集 | [0, 500) | 500 |
| 训练集 | [500, 12087) | ~11,500 |

### 数据加载

与其他任务不同，Webshop 的数据是在环境运行时动态加载的：

```python
from rllm.data.utils import load_webshop_data

# 加载训练数据（生成 goal_idx 索引）
train_data = load_webshop_data("train", num_samples=1000)

# 加载测试数据
test_data = load_webshop_data("test", num_samples=100)
```

## 奖励设计

### 奖励类型

**二元奖励（0 或 1）** - 与其他任务（math, code）保持一致。

- **1.0**: 完美完成任务（task_score == 1.0）
- **0.0**: 未完美完成任务

### 任务评分（task_score）

环境内部会计算一个 0.0 ~ 1.0 的连续匹配分数，基于以下维度：

| 维度 | 说明 | 权重 |
|------|------|------|
| **类型匹配 (r_type)** | 产品类型是否匹配（query、category、title） | 乘法因子 |
| **属性匹配 (r_att)** | 产品属性匹配数 / 目标属性数 | 加权 |
| **选项匹配 (r_option)** | 选项匹配数 / 目标选项数 | 加权 |
| **价格匹配 (r_price)** | 价格是否在预算内 | 加权 |

### 计算公式

```python
# 基础分数（环境内部计算）
base_score = (attr_matches + option_matches + price_match) / (num_attrs + num_options + 1)
task_score = base_score * r_type  # 范围 [0.0, 1.0]

# 最终奖励（二元）
reward = 1.0 if task_score == 1.0 else 0.0
```

### 判定标准

| task_score | 奖励 | 状态 |
|------------|------|------|
| 1.0 | 1.0 | 完美完成（perfect） |
| < 1.0 | 0.0 | 未完成（partial/failed） |

## 使用方法

### 在多任务训练中使用

```bash
python3 -m examples.multi_task.train_composite \
    +data.webshop_num=1000 \
    +data.webshop_path=null \
    # ... 其他参数
```

### 单独使用 Webshop 环境

```python
from rllm.environments.webshop import WebshopEnvironment
from rllm.agents.webshop_agent import WebshopAgent

# 创建环境
env = WebshopEnvironment(
    max_steps=15,
    observation_mode="text",
    webshop_path="/path/to/webshop"  # 可选
)

# 创建 Agent
agent = WebshopAgent()

# 运行一个 episode
obs, info = env.reset(task={"goal_idx": 500})
agent.update_from_env(obs, 0, False, info)

done = False
while not done:
    # 获取 agent 的 chat_completions 并调用 LLM
    response = call_llm(agent.chat_completions)
    
    # 解析动作
    action = agent.update_from_model(response)
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    agent.update_from_env(obs, reward, done, info)

print(f"Task Score: {info['task_score']}")  # 0.0 ~ 1.0
```

## Agent 动作格式

Agent 需要使用以下格式输出动作：

```
<think>分析当前页面和任务需求...</think>
<action>search[red running shoes]</action>
```

或

```
<think>这个产品符合要求，点击购买...</think>
<action>click[Buy Now]</action>
```

### 可用动作

| 动作 | 格式 | 说明 |
|------|------|------|
| 搜索 | `search[keywords]` | 搜索产品 |
| 点击 | `click[element]` | 点击按钮或链接 |

## 文件结构

```
rllm/environments/webshop/
├── __init__.py
├── webshop_env.py      # WebshopEnvironment 类
└── README.md           # 本文档

rllm/agents/
└── webshop_agent.py    # WebshopAgent 类

rllm/rewards/
└── webshop_reward.py   # 奖励函数
```

## 参考

- [WebShop 论文](https://arxiv.org/abs/2207.01206)
- [WebShop GitHub](https://github.com/princeton-nlp/webshop)
- [原始环境实现](../../agent_system/environments/env_package/webshop/)
