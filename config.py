"""
VeilBench Configuration
API 配置、评委配置、评分参数
"""

import os
import random
import re

# 从 .env 文件加载环境变量（.env 不应提交到版本控制）
from dotenv import load_dotenv
load_dotenv()


def _get_api_key(env_var: str) -> str:
    """从环境变量读取 API key，未设置时抛出异常"""
    val = os.getenv(env_var)
    if not val:
        raise RuntimeError(
            f"环境变量 {env_var} 未设置。\n"
            f"请将密钥填入 .env 文件：echo '{env_var}=sk-xxx' >> .env"
        )
    return val


# ==================== API Keys ====================
_DEEPSEEK_API_KEY = _get_api_key("DEEPSEEK_API_KEY")
_SILICONFLOW_API_KEY = _get_api_key("SILICONFLOW_API_KEY")
_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or _DEEPSEEK_API_KEY


# ==================== 待评测模型配置 ====================
# 只测思考模式（reasoning mode）
# DeepSeek: reasoning_effort = "high" / "max"
# Qwen: 默认启用思考模式，无需额外参数
MODELS = {
    "deepseek-v4-flash-high": {
        "name": "deepseek-v4-flash (high)",
        "api_base": "https://api.deepseek.com",
        "api_key": _DEEPSEEK_API_KEY,
        "model_id": "deepseek-v4-flash",
        "thinking_mode": True,
        "reasoning_effort": "high",
    },
    "deepseek-v4-flash-max": {
        "name": "deepseek-v4-flash (max)",
        "api_base": "https://api.deepseek.com",
        "api_key": _DEEPSEEK_API_KEY,
        "model_id": "deepseek-v4-flash",
        "thinking_mode": True,
        "reasoning_effort": "max",
    },
    "deepseek-v4-pro-high": {
        "name": "deepseek-v4-pro (high)",
        "api_base": "https://api.deepseek.com",
        "api_key": _DEEPSEEK_API_KEY,
        "model_id": "deepseek-v4-pro",
        "thinking_mode": True,
        "reasoning_effort": "high",
    },
    "deepseek-v4-pro-max": {
        "name": "deepseek-v4-pro (max)",
        "api_base": "https://api.deepseek.com",
        "api_key": _DEEPSEEK_API_KEY,
        "model_id": "deepseek-v4-pro",
        "thinking_mode": True,
        "reasoning_effort": "max",
    },
    "Qwen3.6-27B": {
        "name": "Qwen/Qwen3.6-27B",
        "api_base": "https://api.siliconflow.cn/v1/",
        "api_key": _SILICONFLOW_API_KEY,
        "model_id": "Qwen/Qwen3.6-27B",
        "thinking_mode": True,
        "reasoning_effort": None,
    },
    "mimo-v2.5-pro": {
        "name": "mimo-v2.5-pro",
        "api_base": "http://127.0.0.1:8317/v1",
        "api_key": "0796",
        "model_id": "mimo-v2.5-pro(xhigh)",
        "thinking_mode": True,
        "reasoning_effort": None,
        "max_tokens": 131072,
    },
    "glm-5.1": {
        "name": "glm-5.1 (xhigh)",
        "api_base": "http://127.0.0.1:8317/v1",
        "api_key": "0796",
        "model_id": "glm-5.1(xhigh)",
        "thinking_mode": True,
        "reasoning_effort": None,
    },
    "kimi-k2.6": {
        "name": "kimi-k2.6 (xhigh)",
        "api_base": "http://127.0.0.1:8317/v1",
        "api_key": "0796",
        "model_id": "kimi-k2.6(xhigh)",
        "thinking_mode": True,
        "reasoning_effort": None,
    },
}


# ==================== 评委模型配置 ====================
# 评委独立于被评测模型，不开思考模式（低温度、高一致性）
JUDGE_API_CONFIG = {
    "api_base": "https://api.deepseek.com",
    "api_key": _JUDGE_API_KEY,
    "model_id": "deepseek-v4-pro",
}

MULTI_JUDGE_ENABLED = False


# ==================== 请求参数 ====================
MAX_TOKENS = 384000   # 384k，包含思考 token
TIMEOUT = 4800        # 80 分钟
REQUEST_INTERVAL = 1  # 模型请求间隔（秒）

# 评委请求参数
JUDGE_MAX_TOKENS = 65536
JUDGE_TEMPERATURE = 0.2


# ==================== 评分参数 ====================
OBJECTIVE_WEIGHT = 0.40
SUBJECTIVE_WEIGHT = 0.60
Z_SCORE_NORMALIZATION = True
JUDGE_DISCREPANCY_THRESHOLD = 2.0


# ==================== 匿名评分配置 ====================
def generate_anon_map(model_keys: list) -> dict:
    """生成模型匿名映射表，每次调用随机打乱"""
    shuffled = model_keys.copy()
    random.shuffle(shuffled)
    return {mk: f"Model_{chr(65 + i)}" for i, mk in enumerate(shuffled)}


COMMON_MODEL_SELF_REFS = [
    "DeepSeek", "deepseek", "Qwen", "qwen", "Kimi", "kimi",
    "Claude", "claude", "GPT", "gpt", "Gemini", "gemini",
    "Mimo", "mimo", "GLM", "glm", "ChatGLM", "chatglm", "智谱",
    "Llama", "llama", "Mistral", "mistral", "通义千问",
]


def anonymize_prompt(judge_prompt: str, anon_map: dict) -> str:
    """替换模型真实 ID 为匿名代号，清除自指词汇"""
    anonymized = judge_prompt
    for mk, aid in anon_map.items():
        patterns = [
            re.escape(mk),
            re.escape(mk.replace("-", " ")),
            re.escape(mk.replace("-", "_")),
            re.escape(mk.replace("/", " ")),
            re.escape(mk.split("/")[-1]) if "/" in mk else re.escape(mk),
        ]
        seen = set()
        for pat in patterns:
            if pat and pat not in seen:
                seen.add(pat)
                anonymized = re.sub(pat, aid, anonymized, flags=re.IGNORECASE)

    for word in COMMON_MODEL_SELF_REFS:
        pattern = rf'(?<![a-zA-Z0-9_\-]){re.escape(word)}(?![a-zA-Z0-9_\-])'
        anonymized = re.sub(pattern, '[MODEL_NAME]', anonymized)

    return anonymized


JUDGE_SYSTEM_PROMPT = """你是一位严格的AI评测专家。你的任务是对大语言模型的回答进行客观、公正的评分。

【绝对规则】
1. 你正在评分的答案是匿名的，标识为 Model_A / Model_B / Model_C 等。你严禁猜测或推断任何模型的真实身份。
2. 严格按照提供的评分标准进行打分，不因模型的表达风格（如简洁/冗长、中文/英文偏好）而偏袒。
3. 每个维度给出 1-10 分的评分，保留 1 位小数。
4. 必须给出具体的评分理由，指出具体哪里好、哪里差。
5. 对于客观题（如数学题、逻辑题），答案错误必须大幅扣分；推导过程错误但答案蒙对，不给满分。
6. 最终输出必须是严格的 JSON 格式。

【输出格式】
```json
{
    "dimension_scores": {
        "维度1": 分数,
        "维度2": 分数,
        ...
    },
    "overall_score": 综合分数,
    "reasoning": "详细的评分理由",
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"]
}
```"""


# ==================== 报告图表颜色 ====================
CHART_COLORS = ["#c9a96e", "#7a8ba3", "#8b6f47", "#5b8fa8", "#a0785a"]
CHART_FILLS = [
    "rgba(201,169,110,0.25)", "rgba(122,139,163,0.25)",
    "rgba(139,111,71,0.25)", "rgba(91,143,168,0.25)", "rgba(160,120,90,0.25)",
]
