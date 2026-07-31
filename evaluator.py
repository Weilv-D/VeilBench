"""
VeilBench Scoring Engine
1. 匿名评分 - 隐藏模型 ID，使用 Model_A/B/C
2. 客观验证集成 - 自动验证 + 主观评分加权
3. z-score 标准化 - 消除不同评委评分尺度偏差
"""

import json
import os
import re
import math
import random
from typing import Dict, List, Tuple
from models import ModelClient
from config import (
    JUDGE_API_CONFIG,
    MULTI_JUDGE_ENABLED,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    Z_SCORE_NORMALIZATION,
    OBJECTIVE_WEIGHT,
    SUBJECTIVE_WEIGHT,
    JUDGE_DISCREPANCY_THRESHOLD,
    generate_anon_map,
    anonymize_prompt,
)
from validators import run_objective_validation


# ==================== 参考答案策略 ====================

# 适合提供参考答案的类别：有确定或半确定答案的题目
CATEGORIES_WITH_REFERENCE = {"math", "algorithm", "system", "debug", "logic"}


def _append_reference(lines: List[str], test_case):
    """在需要时追加参考答案，提示评委仅作参照"""
    if test_case.category not in CATEGORIES_WITH_REFERENCE:
        return
    if not getattr(test_case, "expected_answer", None):
        return
    lines.append("### 参考答案（仅作评分参照，允许等价或更优变体）")
    lines.append(test_case.expected_answer)
    lines.append("")
    lines.append(
        "【评委注意】参考答案展示的是预期解法之一。若模型给出了等价或更优的方案，"
        "应给予同等或更高分数。评分核心应关注推导过程的严谨性、自洽性与完整性，"
        "而非与参考答案的字面一致性。"
    )
    lines.append("")


# ==================== 匿名化层 ====================

def anonymize_answers_for_judge(test_case, model_answers: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """构建匿名化的多答案评分 prompt"""
    model_keys = list(model_answers.keys())
    anon_map = generate_anon_map(model_keys)

    lines = [
        f"## 评测任务：{test_case.name}", "",
        "### 题目信息",
        f"- 题目编号：{test_case.id}",
        f"- 题目类型：{test_case.category}", "",
        "### 原始题目",
        test_case.prompt, "",
    ]

    shuffled = list(anon_map.items())
    random.shuffle(shuffled)
    for mk, aid in shuffled:
        lines.append("---")
        lines.append(f"### {aid} 的回答")
        lines.append(model_answers[mk])
        lines.append("")

    lines.append("### 评分标准")
    lines.append(test_case.evaluation_criteria)
    lines.append("")

    _append_reference(lines, test_case)

    lines.append(
        "请对以上每个模型的回答分别评分。输出格式为 JSON 数组，每个元素对应一个模型：\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "model_id": "Model_A",\n'
        '    "dimension_scores": {...},\n'
        '    "overall_score": 分数,\n'
        '    "reasoning": "...",\n'
        '    "strengths": [...],\n'
        '    "weaknesses": [...]\n'
        "  },\n"
        "  ...\n"
        "]\n"
        "```"
    )

    prompt = "\n".join(lines)
    prompt = anonymize_prompt(prompt, anon_map)
    return prompt, anon_map


# ==================== 响应解析 ====================

def _extract_outermost_array(text: str) -> str:
    """从文本中提取最外层 JSON 数组（正确处理嵌套括号与字符串内字符）"""
    start = text.find("[")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def parse_multi_judge_response(content: str) -> List[dict]:
    """解析评委返回的 JSON 数组"""
    fenced = re.search(r'```json\s*([\s\S]*?)```', content, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else content.strip()
    json_str = _extract_outermost_array(candidate)
    if not json_str:
        json_str = candidate

    try:
        results = json.loads(json_str)
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return [results]
        return []
    except json.JSONDecodeError:
        return []


# ==================== 评分核心 ====================

def run_single_judge(prompt: str, system_prompt: str = None) -> dict:
    """调用评委模型评分"""
    client = ModelClient("judge", judge_config=JUDGE_API_CONFIG)
    print(f"    [API] 调用评委模型 {JUDGE_API_CONFIG['model_id']}...", flush=True)
    return client.generate(
        prompt=prompt,
        system_prompt=system_prompt or JUDGE_SYSTEM_PROMPT,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )


def evaluate_all_models_single_prompt(test_case, model_answers: Dict[str, str]) -> Dict[str, dict]:
    """
    多模型答案匿名共现，评委一次性评分。
    返回: {model_key: eval_result}
    优先使用 results/_judge_cache/<test_id>.json 的评委结果（由 judge_runner 生成），
    避免重复调用评委 API。
    """
    cache_path = os.path.join("results", "_judge_cache", f"{test_case.id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        cached_result = cache.get("response", {})
        cached_complete = bool(cache.get("complete"))
        if cached_result.get("success") and cached_complete:
            print(f"    [CACHE] 使用缓存的评委评分 {test_case.id}", flush=True)
            result = cached_result
            reverse_map = {v: k for k, v in cache["anon_map"].items()}
        else:
            result = None
            reverse_map = None
    else:
        result = None
        reverse_map = None

    if result is None:
        prompt, anon_map = anonymize_answers_for_judge(test_case, model_answers)
        reverse_map = {v: k for k, v in anon_map.items()}
        result = run_single_judge(prompt)

    if not result["success"]:
        return {mk: {
            "success": False, "error": result["error"],
            "scores": {}, "overall": 0,
        } for mk in model_answers}

    parsed_list = parse_multi_judge_response(result["content"])

    if not parsed_list:
        return {mk: {
            "success": False,
            "error": "评委未返回有效 JSON 数组",
            "raw": result["content"][:500],
        } for mk in model_answers}

    results = {}
    for item in parsed_list:
        aid = item.get("model_id", "")
        mk = reverse_map.get(aid)
        if mk is None:
            continue
        results[mk] = {
            "success": True,
            "scores": item.get("dimension_scores", {}),
            "overall": item.get("overall_score", 0),
            "reasoning": item.get("reasoning", ""),
            "strengths": item.get("strengths", []),
            "weaknesses": item.get("weaknesses", []),
            "raw_judge_response": result["content"],
            "anon_id": aid,
            "judge_scores": [{"judge": "single_judge", "overall": item.get("overall_score", 0), "blended": item.get("overall_score", 0)}],
        }

    for mk in model_answers:
        if mk not in results:
            results[mk] = {
                "success": False,
                "error": "评委未返回该模型的评分",
            }

    return results


# ==================== 多评委与标准化 ====================

def evaluate_with_multiple_judges(test_case, model_answers: Dict[str, str], judge_keys: List[str]) -> Dict[str, dict]:
    """多评委交叉验证，返回: {model_key: merged_eval_result}"""
    all_judge_results = []

    for jk in judge_keys:
        print(f"  [JUDGE] Running {jk}...")
        judge_result = evaluate_all_models_single_prompt(test_case, model_answers)
        all_judge_results.append((jk, judge_result))

    merged = {}
    for mk in model_answers:
        judge_scores = []
        judge_details = []

        for jk, jr in all_judge_results:
            ev = jr.get(mk, {})
            if ev.get("success"):
                judge_scores.append({
                    "judge": jk,
                    "overall": ev.get("overall", 0),
                    "blended": ev.get("blended_score", ev.get("overall", 0)),
                })
                judge_details.append({
                    "judge": jk,
                    "reasoning": ev.get("reasoning", ""),
                    "scores": ev.get("scores", {}),
                })

        if not judge_scores:
            merged[mk] = {"success": False, "error": "所有评委评分失败"}
            continue

        if len(judge_scores) >= 3:
            blended_vals = sorted([s["blended"] for s in judge_scores])
            final_blended = blended_vals[len(blended_vals) // 2]
        else:
            final_blended = sum(s["blended"] for s in judge_scores) / len(judge_scores)

        max_diff = max(s["blended"] for s in judge_scores) - min(s["blended"] for s in judge_scores)
        merged[mk] = {
            "success": True,
            "overall": final_blended,
            "blended_score": final_blended,
            "judge_scores": judge_scores,
            "judge_details": judge_details,
            "discrepancy_flag": max_diff > JUDGE_DISCREPANCY_THRESHOLD,
            "max_judge_diff": max_diff,
        }

    return merged


def compute_z_scores(evaluated_results: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """对评委评分进行 z-score 标准化，返回 {model_key: {test_id: z_score}}"""
    if not Z_SCORE_NORMALIZATION:
        return {}

    judge_all_scores = {}
    for mk, results in evaluated_results.items():
        for r in results:
            eval_info = r.get("evaluation", {})
            if not eval_info.get("success"):
                continue
            for js in eval_info.get("judge_scores", []):
                judge_all_scores.setdefault(js["judge"], []).append(js["blended"])

    judge_stats = {}
    for jk, scores in judge_all_scores.items():
        if len(scores) < 2:
            continue
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance) if variance > 0 else 1.0
        judge_stats[jk] = (mean, std)

    z_scored = {}
    for mk, results in evaluated_results.items():
        z_scored[mk] = {}
        for r in results:
            eval_info = r.get("evaluation", {})
            if not eval_info.get("success"):
                continue
            normalized_scores = []
            for js in eval_info.get("judge_scores", []):
                jk = js["judge"]
                if jk in judge_stats:
                    mean, std = judge_stats[jk]
                    z = (js["blended"] - mean) / std
                    normalized = max(0, min(10, 5.0 + z * 2.5))
                    normalized_scores.append(normalized)
            if normalized_scores:
                z_scored[mk][r["test_id"]] = sum(normalized_scores) / len(normalized_scores)

    return z_scored
