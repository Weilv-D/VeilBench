# VeilBench

VeilBench 是一个面向具备思考/推理模式（Thinking Mode）的大语言模型的多维基准评测框架。框架围绕**匿名化共现评分**、**自动化客观验证**与**主客观加权融合**三条核心机制设计，旨在消除 LLM-as-a-Judge 中的身份偏见，并以可复现的方式量化模型在数学推理、算法实现、系统分析、逻辑推演等 8 个维度的能力差异。

---

## 核心设计

### 1. 匿名化共现评分（Anonymized Co-present Evaluation）

多模型答案在提交给评委模型前，会经过两层匿名化处理：

- **标识匿名化**：所有模型真实 ID 被替换为随机打乱的 `Model_A / Model_B / Model_C` 等代号，且每次评测顺序随机。
- **内容匿名化**：回答文本中的模型自指词汇（如 DeepSeek、Qwen、Kimi 等）被全局替换为 `[MODEL_NAME]`，防止评委通过回答风格推断模型身份。

同一道题的所有模型答案会被拼接成一份共现（Co-present）Prompt 发送给评委，使评委在相对比较中打分，进一步降低绝对尺度漂移。

### 2. 自动化客观验证（Objective Validation）

对于存在确定答案或可通过代码执行的题目，系统在本地运行自动化验证，不依赖评委模型的主观判断。当前支持四种验证模式：

| 验证类型 | 适用场景 | 说明 |
|---------|---------|------|
| `exact_text` | 关键词命中 | 检查回答中是否包含必须出现或禁止出现的文本/正则 |
| `math_range` | 数值范围 | 从回答中提取数值（支持 LaTeX `\frac`、百分比、小数），并与预期范围比对；可附加关键词约束 |
| `python_exec` | 代码执行 | 提取回答中的 Python 代码块，在隔离子进程中运行并与预期输出比对 |
| `code_fix` | 代码修复 | 检查回答是否包含修复后的代码模式，同时验证文本层面的修改说明 |

客观验证器具备**场景区分**与**上下文关联**能力：当同一题目存在多组参数场景（如 `p=0.6` 与 `p=0.5`）时，验证器优先在对应参数的局部上下文中提取数值，避免跨场景混淆。

### 3. 主客观加权融合（Blended Scoring）

每道题的最终得分为：

```
Blended = 客观分 × 0.4 + 主观分 × 0.6
```

- 若客观验证通过，则客观分为满分（10 分制）；部分通过时按比例折算。
- 若客观验证不可用（如开放式写作题），则完全依赖主观评分。
- 系统可选启用 **Z-Score 标准化**，对不同评委模型的评分尺度差异进行校正。

---

## 评测维度

VeilBench 目前包含 32 道评测题目，覆盖 8 个维度，每个维度 4 题：

| 维度 | 标识 | 考查重点 |
|------|------|---------|
| 数学推理 | `math` | 概率论、线性代数、博弈论、随机过程的精确推导与计算 |
| 算法实现 | `algorithm` | 数据结构、算法设计与复杂度分析，要求输出可运行代码 |
| 系统设计 | `system` | 高并发、分布式、存储系统的架构设计与权衡分析 |
| 代码调试 | `debug` | 在复杂代码中定位并修复 bug，要求给出根因与修复方案 |
| 长文本写作 | `writing` | 结构化长文生成、多段落逻辑衔接与表达质量 |
| 逻辑推理 | `logic` | 组合逻辑、信息论、博弈策略与归纳推理 |
| 多语言能力 | `multilingual` | 跨语言理解、翻译与文化语境处理 |
| 长上下文 | `long_context` | 超长文本中的信息定位、关联分析与摘要能力 |

---

## 项目结构

```
.
├── config.py           # 项目配置：模型列表、API 参数、评分权重、匿名化规则
├── prompts.py          # 32 道评测题目的 Prompt、期望答案与验证配置
├── models.py           # OpenAI-Compatible API 封装，支持思考模式参数
├── run_benchmark.py    # 阶段 1：增量式回答收集（支持断点续跑）
├── evaluate.py         # 阶段 2：客观验证 + 匿名化主观评分
├── evaluator.py        # LLM-as-a-Judge 引擎：匿名化、JSON 解析、多评委融合、Z-Score
├── report.py           # 阶段 3：生成 HTML 报告（雷达图、热力图、逐题分析）
├── report.py           # 生成 HTML 报告（支持完整报告与仅客观分模式）
├── benchmark.py        # 入口概览工具，打印项目信息与执行指引
├── validators/         # 客观验证引擎
│   └── __init__.py     # exact_text / math_range / python_exec / code_fix 实现
└── results/            # 各模型原始回答（JSON）与评估结果
```

---

## 快速开始

### 环境准备

```bash
# 创建虚拟环境（推荐 Python 3.10+）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install openai
```

### 配置 API 密钥

编辑 `config.py` 中的密钥，或优先通过环境变量注入：

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export SILICONFLOW_API_KEY="your-siliconflow-key"
export JUDGE_API_KEY="your-judge-key"      # 默认为 DEEPSEEK_API_KEY
```

### 三阶段执行

```bash
# 阶段 1：收集回答（按模型逐个执行，支持断点续跑）
python3 run_benchmark.py deepseek-v4-pro-max
python3 run_benchmark.py deepseek-v4-flash-high
# ... 对每个目标模型重复

# 阶段 2：匿名评估（客观验证 + 主观评分）
python3 evaluate.py

# 阶段 3：生成报告
python3 report.py
```

执行完成后，会在项目根目录生成 `report.html`，使用浏览器打开即可查看完整评测报告。

如果只需要快速查看客观验证结果（跳过耗时的主观评分），可运行：

```bash
python3 report.py --objective-only
```

---

## 配置说明

`config.py` 中可调节的核心参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OBJECTIVE_WEIGHT` | `0.40` | 客观分在最终得分中的权重 |
| `SUBJECTIVE_WEIGHT` | `0.60` | 主观分在最终得分中的权重 |
| `Z_SCORE_NORMALIZATION` | `True` | 是否启用评委评分的 Z-Score 标准化 |
| `JUDGE_TEMPERATURE` | `0.2` | 评委模型的生成温度（低温度以保证评分一致性） |
| `MAX_TOKENS` | `384000` | 被评测模型的最大输出 token 数（含思考 token） |

---

## 添加新模型

在 `config.py` 的 `MODELS` 字典中增加条目即可：

```python
"your-model-key": {
    "name": "Display Name",
    "api_base": "https://api.provider.com/v1",
    "api_key": _YOUR_API_KEY,
    "model_id": "provider/model-id",
    "thinking_mode": True,           # 是否启用思考模式
    "reasoning_effort": "high",      # 思考深度参数（如适用）
}
```

---

## 添加新题目

在 `prompts.py` 中创建 `TestCase` 实例并加入对应的测试列表（如 `MATH_TESTS`）。关键字段：

- `id` / `category` / `name`：题目标识与分类
- `prompt`：发送给模型的完整题目文本
- `expected_answer`：预期答案（用于客观验证参考）
- `objective_type`：验证类型（`none` / `exact_text` / `math_range` / `python_exec` / `code_fix`）
- `objective_config`：验证器的详细配置（如数值范围、关键词、正则等）
- `evaluation_criteria`：发送给评委模型的评分标准描述
- `code_to_run`：对于 `python_exec` 类型，提供验证用输入/断言代码

---

## License

MIT
