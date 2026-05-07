#!/usr/bin/env python3
"""
VeilBench — Phase 2: 匿名评分
每道题把所有模型的答案匿名化后，共同呈现给评委
评委只看到 Model_A/B/C...，看不到真实模型 ID
客观验证本地执行，与主观评分加权融合
支持 --models 仅评估指定模型并合并到已有结果
"""

import argparse
import copy
import json
import os
import sys
from config import MODELS, OBJECTIVE_WEIGHT, SUBJECTIVE_WEIGHT
from prompts import ALL_TESTS
from validators import run_objective_validation
from evaluator import evaluate_all_models_single_prompt, compute_z_scores

RESULTS_DIR = "results"


def load_raw_results() -> dict:
    """从各模型单独的结果文件加载"""
    all_results = {}
    for mk in MODELS:
        path = os.path.join(RESULTS_DIR, f"{mk.replace('/', '-')}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_results[mk] = json.load(f)
        else:
            print(f"Warning: No results for {mk}")
            all_results[mk] = []
    return all_results


def _save_checkpoint(evaluated: dict, step: int, output_name: str = "evaluated_results.json"):
    """增量保存评估结果到 results/<output_name>。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, output_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(evaluated, f, ensure_ascii=False, indent=2)
    print(f"  [CHECKPOINT] 已保存至 {path} ({step} 题)", flush=True)


def blend_scores(obj_score, subj_score) -> float:
    """客观分与主观分加权融合"""
    if obj_score is None:
        return subj_score
    return round(OBJECTIVE_WEIGHT * obj_score + SUBJECTIVE_WEIGHT * subj_score, 2)


def main():
    parser = argparse.ArgumentParser(description="VeilBench 匿名评估")
    parser.add_argument("--models", type=str, default=None,
                        help="前置检查：确保指定模型有数据后才开始评估")
    parser.add_argument("--only-complete", action="store_true",
                        help="仅评估所有模型都已完成的题目")
    args = parser.parse_args()

    target_models = set(args.models.split(",")) if args.models else None

    print("=" * 60, flush=True)
    print("VeilBench — Anonymous Evaluation", flush=True)
    print("Mode: Co-presented answers, fully anonymized", flush=True)
    print("=" * 60, flush=True)

    raw_results = load_raw_results()
    # 所有有数据的模型
    all_model_keys = [mk for mk in MODELS if len(raw_results.get(mk, [])) > 0]
    skipped = [mk for mk in MODELS if len(raw_results.get(mk, [])) == 0]
    if skipped:
        print(f"[SKIP] 无数据，跳过: {skipped}", flush=True)

    # --models: 前置检查，确保指定模型有数据
    if target_models:
        missing = target_models - set(all_model_keys)
        if missing:
            print(f"Error: 指定模型无数据: {missing}", flush=True)
            sys.exit(1)

    # 评估所有有数据的模型，共同评分，保证公平比较
    model_keys = all_model_keys
    existing_path = os.path.join(RESULTS_DIR, "evaluated_results.json")

    # --only-complete: 过滤出所有模型都有答案的题目，并跳过已评估的
    tests_to_eval = ALL_TESTS
    if args.only_complete:
        # 加载已有评估结果，避免重复评估
        already_evaluated = set()
        if os.path.exists(existing_path):
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_eval = json.load(f)
            # 已评估的题目：所有模型都有评分结果的 test_id
            for t in ALL_TESTS:
                if all(
                    any(r.get("test_id") == t.id and r.get("evaluation", {}).get("success")
                        for r in existing_eval.get(mk, []))
                    for mk in model_keys
                ):
                    already_evaluated.add(t.id)
            if already_evaluated:
                print(f"[INFO] --only-complete: 已有 {len(already_evaluated)} 道题已评估，跳过", flush=True)

        tests_to_eval = []
        for t in ALL_TESTS:
            if t.id in already_evaluated:
                continue
            if all(
                any(r.get("test_id") == t.id and r.get("success")
                    for r in raw_results.get(mk, []))
                for mk in model_keys
            ):
                tests_to_eval.append(t)
        skipped = len(ALL_TESTS) - len(tests_to_eval) - len(already_evaluated)
        if skipped > 0:
            print(f"[INFO] --only-complete: 跳过 {skipped} 道未全部完成的题目", flush=True)
        if not tests_to_eval:
            print("No new complete tests to evaluate.", flush=True)
            sys.exit(0)
        print(f"[INFO] --only-complete: 新待评估 {len(tests_to_eval)} 道题", flush=True)

    print(f"[INFO] 评估模型: {model_keys} ({len(model_keys)} 个)", flush=True)
    print(f"[INFO] 共 {len(tests_to_eval)} 道题需要评估\n", flush=True)

    # --only-complete: 加载已有评估结果作为基础，新结果追加
    if args.only_complete and os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            evaluated = json.load(f)
        # 确保 model_keys 都有键
        for mk in model_keys:
            if mk not in evaluated:
                evaluated[mk] = []
        print(f"[INFO] 加载已有评估结果，共 {sum(len(v) for v in evaluated.values())} 条记录", flush=True)
    else:
        evaluated = {mk: [] for mk in model_keys}

    for idx, test_case in enumerate(tests_to_eval, 1):
        print(f"\n{'=' * 60}", flush=True)
        print(f"【题目 {idx}/{len(ALL_TESTS)}】{test_case.id} - {test_case.name}", flush=True)
        print(f"类型: {test_case.category} | 客观验证: {test_case.objective_type}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model_answers = {}
        obj_results = {}
        for mk in model_keys:
            for r in raw_results.get(mk, []):
                if r.get("test_id") == test_case.id and r.get("success"):
                    model_answers[mk] = r["answer"]
                    # 优先复用 Phase 1 已保存的客观验证结果
                    cached_obj = r.get("objective_validation")
                    if cached_obj is not None and cached_obj.get("score") is not None:
                        obj_val = cached_obj
                        print(f"  [OBJ] {mk}: {obj_val['score']:.2f} (cached)", flush=True)
                    else:
                        obj_val = run_objective_validation(test_case, r["answer"])
                        if obj_val["score"] is not None:
                            print(f"  [OBJ] {mk}: {obj_val['score']:.2f}", flush=True)
                    obj_results[mk] = obj_val
                    break

        if len(model_answers) < 1:
            print(f"  [SKIP] 无任何有效答案，跳过评分", flush=True)
            for mk in model_keys:
                found = False
                for r in raw_results.get(mk, []):
                    if r.get("test_id") == test_case.id:
                        obj_val = obj_results.get(mk, {})
                        obj_score = obj_val.get("score")
                        rec = copy.deepcopy(r)
                        if obj_score is not None:
                            rec["evaluation"] = {
                                "success": True,
                                "overall": round(obj_score * 10, 2),
                                "blended_score": round(obj_score * 10, 2),
                                "objective_score": round(obj_score * 10, 2),
                                "objective_detail": obj_val.get("detail", ""),
                            }
                        else:
                            rec["evaluation"] = {"success": False, "error": "答案不足或收集失败"}
                        rec["objective_validation"] = obj_val
                        evaluated[mk].append(rec)
                        found = True
                        break
                if not found:
                    evaluated[mk].append({
                        "model": mk, "test_id": test_case.id, "test_name": test_case.name,
                        "category": test_case.category, "success": False,
                        "evaluation": {"success": False, "error": "未找到该题目的原始记录"},
                        "objective_validation": {"score": None, "detail": "未找到原始记录"},
                    })
            continue

        print(f"  [JUDGE] 发送 {len(model_answers)} 个匿名答案共同评分...", flush=True)
        eval_results = evaluate_all_models_single_prompt(test_case, model_answers)

        for mk in model_keys:
            ev = eval_results.get(mk, {})
            obj_val = obj_results.get(mk, {})
            obj_score = obj_val.get("score")
            obj_scaled = obj_score * 10 if obj_score is not None else None

            if ev.get("success"):
                subj_score = ev.get("overall", 0)
                blended = blend_scores(obj_scaled, subj_score)
                ev["blended_score"] = blended
                ev["objective_score"] = round(obj_scaled, 2) if obj_scaled is not None else None
                ev["objective_detail"] = obj_val.get("detail", "")
                obj_str = f"{obj_scaled:.2f}" if obj_scaled is not None else "N/A"
                print(f"  [✓] {mk}: 综合={blended:.2f} | 主观={subj_score:.2f} | 客观={obj_str}", flush=True)
            else:
                if obj_scaled is not None:
                    ev["blended_score"] = round(obj_scaled, 2)
                    ev["objective_score"] = round(obj_scaled, 2)
                    ev["objective_detail"] = obj_val.get("detail", "")
                    ev["success"] = True
                    print(f"  [!] {mk}: 主观失败，使用客观分={obj_scaled:.2f}", flush=True)
                else:
                    print(f"  [✗] {mk}: {ev.get('error', 'Unknown')}", flush=True)

            found = False
            for r in raw_results.get(mk, []):
                if r.get("test_id") == test_case.id:
                    rec = copy.deepcopy(r)
                    rec["evaluation"] = ev
                    rec["objective_validation"] = obj_val
                    evaluated[mk].append(rec)
                    found = True
                    break
            if not found:
                evaluated[mk].append({
                    "model": mk, "test_id": test_case.id, "test_name": test_case.name,
                    "category": test_case.category, "success": False,
                    "evaluation": ev if ev else {"success": False, "error": "未找到该题目的原始记录"},
                    "objective_validation": obj_val if obj_val else {"score": None, "detail": "未找到原始记录"},
                })

        # 每道题完成后增量保存
        _save_checkpoint(evaluated, idx, "evaluated_results.json")

    z_scores = compute_z_scores(evaluated)
    if z_scores:
        print(f"\n[INFO] z-score 标准化完成", flush=True)
        for mk in evaluated:
            for r in evaluated[mk]:
                tid = r.get("test_id")
                if mk in z_scores and tid in z_scores[mk]:
                    r.setdefault("evaluation", {})["z_score"] = round(z_scores[mk][tid], 2)

    # z-score 更新后的最终保存
    _save_checkpoint(evaluated, len(tests_to_eval))
    print(f"\n评分完成！运行 'python3 report.py' 生成报告。", flush=True)


if __name__ == "__main__":
    main()
