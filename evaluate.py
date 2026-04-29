#!/usr/bin/env python3
"""
VeilBench — Phase 2: 匿名评分
每道题把所有模型的答案匿名化后，共同呈现给评委
评委只看到 Model_A/B/C...，看不到真实模型 ID
客观验证本地执行，与主观评分加权融合
"""

import copy
import json
import os
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


def blend_scores(obj_score, subj_score) -> float:
    """客观分与主观分加权融合"""
    if obj_score is None:
        return subj_score
    return round(OBJECTIVE_WEIGHT * obj_score + SUBJECTIVE_WEIGHT * subj_score, 2)


def main():
    print("=" * 60, flush=True)
    print("VeilBench — Anonymous Evaluation", flush=True)
    print("Mode: Co-presented answers, fully anonymized", flush=True)
    print("=" * 60, flush=True)

    raw_results = load_raw_results()
    # 只评估有数据的模型
    model_keys = [mk for mk in MODELS if len(raw_results.get(mk, [])) > 0]
    skipped = [mk for mk in MODELS if len(raw_results.get(mk, [])) == 0]
    if skipped:
        print(f"[SKIP] 无数据，跳过: {skipped}", flush=True)
    print(f"[INFO] 评估模型: {model_keys} ({len(model_keys)} 个)", flush=True)
    print(f"[INFO] 共 {len(ALL_TESTS)} 道题需要评估\n", flush=True)
    evaluated = {mk: [] for mk in model_keys}

    for idx, test_case in enumerate(ALL_TESTS, 1):
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

    z_scores = compute_z_scores(evaluated)
    if z_scores:
        print(f"\n[INFO] z-score 标准化完成", flush=True)
        for mk in evaluated:
            for r in evaluated[mk]:
                tid = r.get("test_id")
                if mk in z_scores and tid in z_scores[mk]:
                    r.setdefault("evaluation", {})["z_score"] = round(z_scores[mk][tid], 2)

    path = os.path.join(RESULTS_DIR, "evaluated_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(evaluated, f, ensure_ascii=False, indent=2)
    print(f"\n评分完成！结果保存至 {path}", flush=True)
    print("运行 'python3 report.py' 生成报告。", flush=True)


if __name__ == "__main__":
    main()
