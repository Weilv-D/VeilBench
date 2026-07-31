#!/usr/bin/env python3
"""
VeilBench — 主观评分执行器（供 subagent 调用）
为指定 test_id 构建匿名评委 prompt、调用评委模型，并将完整结果缓存到
results/_judge_cache/<test_id>.json，避免大体积文本进入主会话上下文。

用法: python3 judge_runner.py <test_id> [<test_id> ...]
"""

import json
import os
import sys
import time

from config import MODELS
from evaluator import anonymize_answers_for_judge, parse_multi_judge_response, run_single_judge
from prompts import ALL_TESTS

CACHE_DIR = os.path.join("results", "_judge_cache")
RESULTS_DIR = "results"
MAX_RETRIES = 3
RETRY_DELAY = 15


def _load_raw_results() -> dict:
    all_results = {}
    for mk in MODELS:
        path = os.path.join(RESULTS_DIR, f"{mk.replace('/', '-')}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_results[mk] = json.load(f)
    return all_results


def _test_by_id(test_id: str):
    for t in ALL_TESTS:
        if t.id == test_id:
            return t
    return None


def run_one(test_id: str, raw_results: dict) -> dict:
    test_case = _test_by_id(test_id)
    if test_case is None:
        return {"test_id": test_id, "success": False, "error": f"未知 test_id: {test_id}"}

    cache_path = os.path.join(CACHE_DIR, f"{test_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("complete"):
            print(f"[SKIP] {test_id} 已缓存，跳过", flush=True)
            return {"test_id": test_id, "success": True, "cached": True}

    model_answers = {}
    for mk, results in raw_results.items():
        for r in results:
            if r.get("test_id") == test_id and r.get("success"):
                model_answers[mk] = r["answer"]
                break

    if not model_answers:
        return {"test_id": test_id, "success": False, "error": "无有效答案可评分"}

    final = None
    for attempt in range(1, MAX_RETRIES + 1):
        prompt, anon_map = anonymize_answers_for_judge(test_case, model_answers)
        result = run_single_judge(prompt)
        if not result["success"]:
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {test_id}: {result['error']}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            continue

        parsed = parse_multi_judge_response(result["content"])
        reverse_map = {v: k for k, v in anon_map.items()}
        covered = {reverse_map.get(item.get("model_id", "")) for item in parsed}
        missing = set(model_answers) - covered
        if not missing:
            final = (result, anon_map, parsed, reverse_map)
            break
        print(
            f"  [INCOMPLETE {attempt}/{MAX_RETRIES}] {test_id}: 仅覆盖 {len(covered)}/{len(model_answers)} 个模型"
            f"，缺失 {sorted(missing)}，重试",
            flush=True,
        )
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    if final is None:
        return {"test_id": test_id, "success": False, "error": "评委评分不完整（多次尝试后仍缺失部分模型）"}

    result, anon_map, parsed, reverse_map = final
    summary = {
        "test_id": test_id,
        "complete": True,
        "model_keys": list(model_answers.keys()),
        "anon_map": anon_map,
        "parsed": parsed,
        "response": result,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    scores = []
    for item in parsed:
        scores.append(f"{item.get('model_id', '?')}={item.get('overall_score', '?')}")
    print(
        f"[OK] {test_id} ({test_case.name}) {result['completion_tokens']} tok, "
        f"{result['elapsed_time']:.0f}s, {len(parsed)}/{len(model_answers)} 评分: {'; '.join(scores)}",
        flush=True,
    )
    return {"test_id": test_id, "success": True, "n_models": len(model_answers)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 judge_runner.py <test_id> [<test_id> ...]")
        sys.exit(1)

    test_ids = sys.argv[1:]
    raw = _load_raw_results()

    ok = 0
    failed = 0
    for tid in test_ids:
        ret = run_one(tid, raw)
        if ret["success"]:
            ok += 1
        else:
            failed += 1
            print(f"[FAIL] {tid}: {ret.get('error')}", flush=True)

    print(f"[DONE] 完成 {ok} 题, 失败 {failed} 题", flush=True)
    if failed > 0:
        sys.exit(1)
