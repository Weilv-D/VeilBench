# VeilBench

<p align="center">
  <b>面向思考模式大语言模型的匿名化多维基准评测框架</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Tests-32%20Problems-orange" alt="32 Problems">
  <img src="https://img.shields.io/badge/Dimensions-8-red" alt="8 Dimensions">
</p>

---

## 简介

VeilBench 是一套专为 **Thinking Mode / Reasoning Mode** 大语言模型设计的基准评测框架。它通过三层机制解决 LLM-as-a-Judge 的核心痛点：

| 痛点 | VeilBench 的解决方案 |
|------|---------------------|
| 评委知道模型身份，产生品牌偏见 | **双层匿名化**：随机代号 + 内容脱敏 |
| 评委评分尺度漂移，难以横向对比 | **共现评分**：同题多答案一起呈现，强制相对比较 |
| 主观评分不稳定，可复现性差 | **客观验证器**：数学/代码/文本本地自动判分 |
| 不同评委模型打分标准不一致 | **Z-Score 标准化**：校正评分尺度差异 |

---

## 效果预览

执行 `python3 report.py` 后生成自包含 HTML 报告，包含：

- **雷达图**：8 维度能力全景对比
- **柱状图**：逐维度得分 breakdown
- **热力图**：模型 × 维度的得分矩阵
- **排行榜**：Overall / Median / StdDev / Speed / Code Pass Rate
- **逐题分析**：每道题的得分、评委理由、优缺点标签
- **CSV 导出**：一键下载排行榜数据

报告采用自适应配色与响应式布局，支持明暗模式自动切换。

---

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/Weilv-D/VeilBench.git
cd VeilBench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # openai, python-dotenv
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的密钥
```

```bash
# .env
DEEPSEEK_API_KEY=sk-xxx
SILICONFLOW_API_KEY=sk-yyy
JUDGE_API_KEY=sk-zzz   # 可选，默认复用 DEEPSEEK
```

> 密钥仅保存在本地 `.env` 文件，已通过 `.gitignore` 排除，不会意外提交。

### 3. 查看项目状态

```bash
python3 benchmark.py
```

输出示例：

```
============================================================
VeilBench — Thinking Mode Benchmark
============================================================

  评测题目: 32 道, 覆盖 8 个维度
     • math            4 题
     • algorithm       4 题
     ...

  目标模型: 5 个
     • deepseek-v4-flash-high      (high)
     ...

  阶段 1: 收集回答
     → python3 run_benchmark.py deepseek-v4-flash-high
     → python3 run_benchmark.py deepseek-v4-pro-max
     ...
```

### 4. 三阶段评测

```bash
# 阶段 1：收集回答（支持断点续跑）
python3 run_benchmark.py deepseek-v4-pro-max
python3 run_benchmark.py Qwen3.6-27B

# 阶段 2：匿名评估（客观验证 + 主观评分）
python3 evaluate.py

# 阶段 3：生成报告
python3 report.py
```

报告默认输出为 `report.html`，浏览器打开即可。

> **仅客观分快速报告**：`python3 report.py --objective-only`（跳过耗时评委调用）

---

## 评测维度

32 道原创题目，覆盖 8 个维度，每个维度 4 题：

| 维度 | 标识 | 典型题目 |
|------|------|---------|
| **数学推理** | `math` | 纳什均衡推导、贝叶斯更新链、随机游走停时 |
| **算法设计** | `algorithm` | 流式基数估算、一致性哈希优化、增量外部排序 |
| **系统架构** | `system` | 10 万并发白板、金融分布式事务、全球多活缓存 |
| **代码调试** | `debug` | 弱引用内存泄漏、asyncio 竞态、SQL 幻读、闭包延迟绑定 |
| **创意写作** | `writing` | 博尔赫斯风格量子计算、五视角叙事、经济学隐喻爱情 |
| **逻辑推理** | `logic` | 囚徒问题扩展、自指悖论、贝叶斯博弈、信息论通信极限 |
| **多语言能力** | `multilingual` | 跨语言语义旅行、古诗词逻辑省略、方言语法对比、歧义消解 |
| **长上下文** | `long_context` | 12 人谋杀案推理、微服务依赖诊断、法律合同冲突、学术论文审稿 |

---

## 核心机制详解

### 匿名化共现评分

```
模型 A 回答 →  Model_X
模型 B 回答 →  Model_Y   ──→  共现 Prompt ──→ 评委模型
模型 C 回答 →  Model_Z
```

1. **标识匿名化**：`generate_anon_map()` 每次随机打乱，映射为 `Model_A/B/C`
2. **内容匿名化**：`anonymize_prompt()` 用正则全局替换 `DeepSeek/Qwen/Kimi` 等自指词为 `[MODEL_NAME]`
3. **共现呈现**：同题所有答案拼接为一份 Prompt，评委在相对比较中打分

### 客观验证器

| 类型 | 能力 | 示例 |
|------|------|------|
| `exact_text` | 关键词/正则匹配 | 检查回答是否包含"MVCC""快照读" |
| `math_range` | 智能数值提取 | 从文本中提取 `\frac`、`%`、小数，按场景（如 p=0.6 vs p=0.5）分组校验 |
| `python_exec` | 沙箱代码执行 | 提取代码块 + 测试代码合并运行，验证断言 |
| `code_fix` | 代码模式检查 | 检查修复是否包含 `try/finally`、是否遗漏 `weakref.ref` 等 |

### 评分公式

```
Blended Score = Objective × 0.4 + Subjective × 0.6
```

- 客观分由本地验证器自动计算（0-10 分制）
- 主观分由匿名评委模型给出（0-10 分制）
- 若某模型主观评分失败，fallback 到纯客观分
- 可选启用 `Z_SCORE_NORMALIZATION` 校正不同评委的尺度偏差

---

## 项目结构

```
.
├── config.py              # 模型配置、API 参数、评分权重、匿名化规则
├── prompts.py             # 32 道 TestCase（prompt / 答案 / 验证配置）
├── models.py              # OpenAI-Compatible API 封装，自动处理 thinking mode
├── run_benchmark.py       # 阶段 1：增量式回答收集（断点续跑）
├── evaluate.py            # 阶段 2：客观验证 + 匿名共现评分
├── evaluator.py           # Judge 引擎：匿名化、JSON 解析、多评委融合、Z-Score
├── report.py              # 阶段 3：HTML 报告生成（雷达/柱状/热力图）
├── benchmark.py           # 入口概览：状态检查 + 执行指引
├── validators/
│   └── __init__.py        # 客观验证引擎实现
├── .env.example           # API 密钥模板（复制为 .env 后填入）
├── .gitignore             # 已排除 .env / results / *.html
└── results/               # 各模型结果 + evaluated_results.json
```

---

## 扩展指南

### 添加新模型

编辑 `config.py` 的 `MODELS` 字典：

```python
"my-model": {
    "name": "My Model (high)",
    "api_base": "https://api.provider.com/v1",
    "api_key": _get_api_key("MY_API_KEY"),   # 在 .env 中添加 MY_API_KEY=sk-xxx
    "model_id": "provider/model-id",
    "thinking_mode": True,
    "reasoning_effort": "high",   # DeepSeek 风格，或设为 None
}
```

### 添加新题目

在 `prompts.py` 中创建 `TestCase`：

```python
TestCase(
    id="X1",
    category="math",
    name="题目名称",
    prompt="...",
    expected_answer="...",
    objective_type="math_range",   # none / exact_text / math_range / python_exec / code_fix
    objective_config={
        "tolerance": 0.1,
        "key_values": {"answer": (3.14, 3.15)},
    },
    evaluation_criteria="...",
    code_to_run=None,   # python_exec 类型时填入测试代码
)
```

---

## 配置参数

`config.py` 中可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OBJECTIVE_WEIGHT` | `0.40` | 客观分权重 |
| `SUBJECTIVE_WEIGHT` | `0.60` | 主观分权重 |
| `JUDGE_TEMPERATURE` | `0.2` | 评委模型温度（低=稳定） |
| `JUDGE_MAX_TOKENS` | `65536` | 评委输出上限 |
| `MAX_TOKENS` | `384000` | 被测模型输出上限（含思考 token） |
| `TIMEOUT` | `4800` | 请求超时（秒） |
| `REQUEST_INTERVAL` | `1` | 模型请求间隔（秒） |
| `Z_SCORE_NORMALIZATION` | `True` | 是否启用 Z-Score 标准化 |

---

## 依赖

- Python 3.10+
- `openai`
- `python-dotenv`

```bash
pip install openai python-dotenv
```

---

## License

MIT
