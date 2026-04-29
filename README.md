# VeilBench

匿名化多维基准评测框架，专为 Thinking Mode / Reasoning Mode 大语言模型设计。

**核心机制**：双层匿名化（标识+内容脱敏）→ 共现评分 → 客观验证（40%）+ 主观评分（60%）→ Z-Score 标准化

---

## 快速开始

```bash
git clone https://github.com/Weilv-D/VeilBench.git
cd VeilBench
python3 -m venv .venv && source .venv/bin/activate
pip install openai python-dotenv
```

配置密钥：
```bash
cp .env.example .env
# 编辑 .env 填入 API key
```

三阶段执行：
```bash
# 1. 收集回答（支持断点续跑）
python3 run_benchmark.py deepseek-v4-pro-max

# 2. 匿名评估
python3 evaluate.py

# 3. 生成报告
python3 report.py            # 完整报告
python3 report.py --objective-only   # 仅客观分（跳过评委）
```

查看状态：`python3 benchmark.py`

---

## 评测维度

8 个维度 × 4 题 = 32 题：数学推理、算法设计、系统架构、代码调试、创意写作、逻辑推理、多语言理解、长上下文。

---

## 项目结构

```
config.py           # 模型配置、API 参数、评分权重
prompts.py          # 32 道评测题目（未提交到仓库）
models.py           # OpenAI-Compatible API 封装
run_benchmark.py    # 阶段 1：增量收集回答
evaluate.py         # 阶段 2：客观验证 + 匿名评分
evaluator.py        # Judge 引擎：匿名化、JSON 解析、Z-Score
report.py           # 阶段 3：HTML 报告（雷达/柱状/热力图）
benchmark.py        # 入口概览：状态检查 + 执行指引
validators/         # 客观验证引擎
```

---

## 扩展

**添加模型**：编辑 `config.py` 的 `MODELS` 字典，在 `.env` 中添加对应 API key。

**添加题目**：在 `prompts.py` 中创建 `TestCase`，指定 `objective_type`（`exact_text` / `math_range` / `python_exec` / `code_fix` / `none`）和验证配置。

---

## License

MIT
