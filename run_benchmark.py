#!/usr/bin/env python3
"""
VeilBench — Phase 1: 增量式回答收集
每个模型结果保存到单独文件，支持断点续跑
"""

import json
import os
import time
import sys

from config import MODELS, MAX_TOKENS, REQUEST_INTERVAL
from models import ModelClient
from prompts import ALL_TESTS
from validators import extract_code_blocks, run_code_verification, run_objective_validation

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _path(model_key):
    return os.path.join(RESULTS_DIR, f"{model_key.replace('/', '-')}.json")


def _load(model_key):
    p = _path(model_key)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def _save(model_key, data):
    with open(_path(model_key), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


def run_model(model_key: str):
    """收集单个模型的全部回答"""
    client = ModelClient(model_key)
    results = _load(model_key)
    done_ids = {r["test_id"] for r in results if r.get("success")}
    # 移除之前失败的记录，以便重试
    results = [r for r in results if r.get("success")]

    for test in ALL_TESTS:
        if test.id in done_ids:
            print(f"[SKIP] {model_key} - {test.id}", flush=True)
            continue

        print(f"[RUN] {model_key} - {test.id}: {test.name}", flush=True)

        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            result = client.generate(prompt=test.prompt, max_tokens=MAX_TOKENS)
            if result["success"]:
                break
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {model_key} - {test.id}: {result['error']}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        if not result["success"]:
            print(f"[FAIL] {model_key} - {test.id}: {result['error']}", flush=True)
            results.append({"model": model_key, "test_id": test.id, "test_name": test.name, "category": test.category, "success": False, "error": result["error"]})
            _save(model_key, results)
            continue

        print(f"[OK] {model_key} - {test.id} ({result['completion_tokens']} tok, {result['elapsed_time']:.1f}s)", flush=True)
        answer = result["content"]

        code_verification = None
        if test.category in ("algorithm", "debug") and test.code_to_run:
            code_blocks = extract_code_blocks(answer)
            if code_blocks:
                code_verification = run_code_verification(code_blocks[0], test.code_to_run)
                status = "PASS" if code_verification["success"] else "FAIL"
                print(f"[CODE] {status}: {code_verification['stdout'].strip()[:80]}", flush=True)

        objective_validation = None
        if test.objective_type != "none":
            objective_validation = run_objective_validation(test, answer)
            if objective_validation["score"] is not None:
                print(f"[OBJ] score={objective_validation['score']:.2f}", flush=True)

        results.append({
            "model": model_key,
            "test_id": test.id,
            "category": test.category,
            "test_name": test.name,
            "success": True,
            "answer": answer,
            "elapsed_time": result["elapsed_time"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "code_verification": code_verification,
            "objective_validation": objective_validation,
        })
        _save(model_key, results)
        time.sleep(REQUEST_INTERVAL)

    print(f"[DONE] {model_key} - first pass complete", flush=True)

    # 重跑失败的题目（最多重试一轮）
    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"[RETRY] {model_key} - {len(failed)} failed tests, retrying...", flush=True)
        failed_ids = {r["test_id"] for r in failed}
        # 从 results 中移除失败记录
        results = [r for r in results if r.get("success")]

        for test in ALL_TESTS:
            if test.id not in failed_ids:
                continue

            print(f"[RETRY] {model_key} - {test.id}: {test.name}", flush=True)

            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                result = client.generate(prompt=test.prompt, max_tokens=MAX_TOKENS)
                if result["success"]:
                    break
                print(f"[RETRY-RETRY {attempt}/{MAX_RETRIES}] {model_key} - {test.id}: {result['error']}", flush=True)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

            if not result["success"]:
                print(f"[FAIL-FINAL] {model_key} - {test.id}: {result['error']}", flush=True)
                results.append({"model": model_key, "test_id": test.id, "test_name": test.name, "category": test.category, "success": False, "error": result["error"]})
                _save(model_key, results)
                continue

            print(f"[OK] {model_key} - {test.id} ({result['completion_tokens']} tok, {result['elapsed_time']:.1f}s)", flush=True)
            answer = result["content"]

            code_verification = None
            if test.category in ("algorithm", "debug") and test.code_to_run:
                code_blocks = extract_code_blocks(answer)
                if code_blocks:
                    code_verification = run_code_verification(code_blocks[0], test.code_to_run)
                    print(f"[CODE] {'PASS' if code_verification['success'] else 'FAIL'}: {code_verification['stdout'].strip()[:80]}", flush=True)

            objective_validation = None
            if test.objective_type != "none":
                objective_validation = run_objective_validation(test, answer)
                if objective_validation["score"] is not None:
                    print(f"[OBJ] score={objective_validation['score']:.2f}", flush=True)

            results.append({
                "model": model_key,
                "test_id": test.id,
                "category": test.category,
                "test_name": test.name,
                "success": True,
                "answer": answer,
                "elapsed_time": result["elapsed_time"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "code_verification": code_verification,
                "objective_validation": objective_validation,
            })
            _save(model_key, results)
            time.sleep(REQUEST_INTERVAL)

        final_failed = sum(1 for r in results if not r.get("success"))
        print(f"[DONE] {model_key} - {final_failed} still failed after retry", flush=True)
    else:
        print(f"[DONE] {model_key} - all {len(results)} tests passed", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_benchmark.py <model_key>")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    model_key = sys.argv[1]
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        sys.exit(1)

    run_model(model_key)
