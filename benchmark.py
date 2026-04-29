#!/usr/bin/env python3
"""
VeilBench — 项目状态概览与执行指引
"""

import json
import os
import sys

from config import MODELS
from prompts import ALL_TESTS

RESULTS_DIR = "results"


def _load_results(model_key: str):
    path = os.path.join(RESULTS_DIR, f"{model_key.replace('/', '-')}.json")
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("VeilBench — Thinking Mode Benchmark")
    print("=" * 60)

    print(f"\n📋 评测题目: {len(ALL_TESTS)} 道, 覆盖 8 个维度")
    for cat in ["math", "algorithm", "system", "debug", "writing", "logic", "multilingual", "long_context"]:
        count = len([t for t in ALL_TESTS if t.category == cat])
        print(f"   • {cat:12s} {count} 题")

    print(f"\n🤖 目标模型: {len(MODELS)} 个")
    for mk, info in MODELS.items():
        mode = info.get("reasoning_effort")
        mode_str = f" ({mode})" if mode else ""
        print(f"   • {mk:30s}{mode_str}")

    print(f"\n📁 结果目录: {RESULTS_DIR}/")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    any_collected = False
    for mk in MODELS:
        results = _load_results(mk)
        success = len([r for r in results if r.get("success")])
        total = len(results)
        status = f"{success}/{len(ALL_TESTS)} 完成"
        if success == 0:
            status = "未开始"
        elif success < len(ALL_TESTS):
            status += " (可断点续跑)"
            any_collected = True
        else:
            status += " ✓"
            any_collected = True
        print(f"   • {mk:30s} {status}")

    evaluated = os.path.exists(os.path.join(RESULTS_DIR, "evaluated_results.json"))
    report = os.path.exists("report.html")

    print("\n" + "-" * 40)
    print("执行指引")
    print("-" * 40)

    if not any_collected:
        print("\n  阶段 1: 收集回答")
        for mk in MODELS:
            print(f"    → python3 run_benchmark.py {mk}")
    else:
        pending = [mk for mk in MODELS
                   if len([r for r in _load_results(mk) if r.get("success")]) < len(ALL_TESTS)]
        if pending:
            print("\n  阶段 1: 继续收集回答")
            for mk in pending:
                print(f"    → python3 run_benchmark.py {mk}")
        else:
            print("\n  阶段 1: 全部模型回答已收集 ✓")

    if not evaluated:
        if any_collected:
            print("\n  阶段 2: 匿名评估")
            print("    → python3 evaluate.py")
    else:
        print("\n  阶段 2: 评估已完成 ✓")

    if not report:
        if evaluated:
            print("\n  阶段 3: 生成报告")
            print("    → python3 report.py")
    else:
        print("\n  阶段 3: 报告已生成 ✓  (report.html)")

    if any_collected and not evaluated:
        print("\n  快捷报告 (仅客观分):")
        print("    → python3 report.py --objective-only")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
