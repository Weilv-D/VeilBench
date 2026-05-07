#!/usr/bin/env python3
"""
VeilBench — Report Generator (Polished)
支持任意数量模型的对比报告，自适应图表与颜色生成。

用法:
    python3 report.py                          # 生成完整报告（需 evaluate.py 结果）
    python3 report.py --objective-only         # 仅基于客观验证生成报告
    python3 report.py --models m1,m2           # 只对比指定模型
    python3 report.py --output my_report.html  # 指定输出路径
"""

import argparse
import csv
import io
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import CHART_COLORS, CHART_FILLS, MODELS
from prompts import ALL_TESTS

# ==================== Constants & Config ====================

RESULTS_DIR = "results"
DEFAULT_REPORT_PATH = "report.html"
MAX_REASONING_PREVIEW = 380

# Category configuration: order, display name, icon
CATEGORY_META = {
    "math":         {"name": "Math",          "icon": "∑", "order": 0},
    "algorithm":    {"name": "Algorithm",     "icon": "⚡", "order": 1},
    "system":       {"name": "System",        "icon": "◈", "order": 2},
    "debug":        {"name": "Debug",         "icon": "🔧", "order": 3},
    "writing":      {"name": "Writing",       "icon": "✎", "order": 4},
    "logic":        {"name": "Logic",         "icon": "◇", "order": 5},
    "multilingual": {"name": "Multilingual",  "icon": "🌐", "order": 6},
    "long_context": {"name": "Long Context",  "icon": "⛋", "order": 7},
}

CATEGORY_ORDER = sorted(CATEGORY_META.keys(), key=lambda c: CATEGORY_META[c]["order"])
CATEGORY_NAMES = {c: CATEGORY_META[c]["name"] for c in CATEGORY_META}
# ==================== Utilities ====================

def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid] + s[mid - 1]) / 2 if n % 2 == 0 else s[mid]


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _avg(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _safe_name(mk: str) -> str:
    return MODELS[mk]["name"] if mk in MODELS else mk


def _load_json(filename: str) -> Any:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== Color Palette ====================

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """HSL (0-360, 0-100, 0-100) → hex"""
    h = h % 360
    s, l = s / 100.0, l / 100.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    segments = [
        (0, 60, c, x, 0), (60, 120, x, c, 0), (120, 180, 0, c, x),
        (180, 240, 0, x, c), (240, 300, x, 0, c), (300, 360, c, 0, x),
    ]
    for lo, hi, r, g, b in segments:
        if lo <= h < hi:
            return f"#{int((r + m) * 255):02x}{int((g + m) * 255):02x}{int((b + m) * 255):02x}"
    return f"#{int(m * 255):02x}{int(m * 255):02x}{int(m * 255):02x}"


def generate_palette(n: int) -> Tuple[List[str], List[str]]:
    """为 n 个模型生成不重复的颜色与填充色。"""
    if n <= len(CHART_COLORS):
        return CHART_COLORS[:n], CHART_FILLS[:n]

    colors, fills = [], []
    golden_angle = 137.508
    for i in range(n):
        h = (i * golden_angle) % 360
        s = 45 + (i % 3) * 8
        l = 38 + (i % 2) * 10
        hex_color = _hsl_to_hex(h, s, l)
        colors.append(hex_color)
        fill_hex = _hsl_to_hex(h, max(s - 25, 15), min(l + 30, 80))
        fr = int(fill_hex[1:3], 16)
        fg = int(fill_hex[3:5], 16)
        fb = int(fill_hex[5:7], 16)
        fills.append(f"rgba({fr},{fg},{fb},0.22)")
    return colors, fills


# ==================== Data Models ====================

@dataclass
class TestDetail:
    test_id: str
    test_name: str
    category: str
    overall: float
    objective_score: Optional[float]
    subjective_score: Optional[float]
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    elapsed_time: float
    completion_tokens: int
    code_passed: Optional[bool]


@dataclass
class ModelStats:
    overall_scores: List[float] = field(default_factory=list)
    details: List[TestDetail] = field(default_factory=list)
    speed_samples: List[float] = field(default_factory=list)
    token_total: int = 0
    code_passed: int = 0
    code_total: int = 0
    cat_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    obj_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    subj_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def avg_overall(self) -> float:
        return _avg(self.overall_scores)

    @property
    def median_overall(self) -> float:
        return _median(self.overall_scores)

    @property
    def std_overall(self) -> float:
        return _std(self.overall_scores)

    @property
    def avg_speed(self) -> float:
        return _avg(self.speed_samples)

    @property
    def code_rate(self) -> float:
        return (self.code_passed / self.code_total * 100) if self.code_total > 0 else 0.0

    def cat_avg(self, cat: str) -> float:
        return _avg(self.cat_scores.get(cat, []))

    def best_cat(self) -> Tuple[str, float]:
        avgs = {c: self.cat_avg(c) for c in CATEGORY_ORDER}
        best = max(avgs, key=avgs.get) if any(avgs.values()) else ("", 0.0)
        return best, avgs.get(best, 0.0)

    def worst_cat(self) -> Tuple[str, float]:
        avgs = {c: self.cat_avg(c) for c in CATEGORY_ORDER}
        worst = min(avgs, key=avgs.get) if any(avgs.values()) else ("", 0.0)
        return worst, avgs.get(worst, 0.0)


# ==================== Data Loading ====================

def load_evaluated_data() -> Dict[str, Any]:
    return _load_json("evaluated_results.json")


def load_objective_only_data() -> Dict[str, Any]:
    all_data = {}
    for mk in MODELS:
        fname = mk.replace("/", "-") + ".json"
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            all_data[mk] = _load_json(fname)
        else:
            all_data[mk] = []
    return all_data


def build_from_evaluated(results: Dict[str, Any], selected: Optional[List[str]]) -> Dict[str, ModelStats]:
    model_keys = list(results.keys())
    if selected:
        model_keys = [mk for mk in model_keys if mk in selected]

    stats: Dict[str, ModelStats] = defaultdict(ModelStats)

    for mk in model_keys:
        ms = stats[mk]
        for r in results.get(mk, []):
            if not r.get("success"):
                continue
            ev = r.get("evaluation", {})
            if not ev.get("success"):
                continue
            cat = r["category"]
            overall = ev.get("blended_score", ev.get("overall", 0))
            ms.overall_scores.append(overall)
            ms.cat_scores[cat].append(overall)

            ms.details.append(TestDetail(
                test_id=r["test_id"],
                test_name=r["test_name"],
                category=cat,
                overall=overall,
                objective_score=ev.get("objective_score"),
                subjective_score=ev.get("overall", 0) if ev.get("success") else None,
                reasoning=ev.get("reasoning", ""),
                strengths=ev.get("strengths", []),
                weaknesses=ev.get("weaknesses", []),
                elapsed_time=r.get("elapsed_time", 0),
                completion_tokens=r.get("completion_tokens", 0),
                code_passed=(r.get("code_verification") or {}).get("success"),
            ))
            if r.get("elapsed_time", 0) > 0:
                ms.speed_samples.append(r["elapsed_time"])
            if r.get("total_tokens", 0) > 0:
                ms.token_total += r["total_tokens"]
            if r.get("code_verification"):
                ms.code_total += 1
                if r["code_verification"]["success"]:
                    ms.code_passed += 1
            obj_s = ev.get("objective_score")
            if obj_s is not None:
                ms.obj_scores[cat].append(obj_s)
            subj = ev.get("overall", 0)
            if ev.get("success"):
                ms.subj_scores[cat].append(subj)

    return dict(stats)


def build_from_objective(raw_data: Dict[str, Any], selected: Optional[List[str]]) -> Dict[str, ModelStats]:
    model_keys = list(MODELS.keys())
    if selected:
        model_keys = [mk for mk in model_keys if mk in selected and mk in raw_data]
    else:
        model_keys = [mk for mk in model_keys if mk in raw_data]

    stats: Dict[str, ModelStats] = defaultdict(ModelStats)

    for mk in model_keys:
        ms = stats[mk]
        for r in raw_data.get(mk, []):
            if not r.get("success"):
                continue
            cat = r["category"]
            obj_val = r.get("objective_validation") or {}
            obj_score = obj_val.get("score")
            if obj_score is not None:
                overall = obj_score * 10
                ms.overall_scores.append(overall)
                ms.cat_scores[cat].append(overall)
                ms.obj_scores[cat].append(overall)
            else:
                overall = 0.0

            ms.details.append(TestDetail(
                test_id=r["test_id"],
                test_name=r["test_name"],
                category=cat,
                overall=overall,
                objective_score=(obj_score * 10) if obj_score is not None else None,
                subjective_score=None,
                reasoning=obj_val.get("detail", ""),
                strengths=[],
                weaknesses=[],
                elapsed_time=r.get("elapsed_time", 0),
                completion_tokens=r.get("completion_tokens", 0),
                code_passed=(r.get("code_verification") or {}).get("success"),
            ))
            if r.get("elapsed_time", 0) > 0:
                ms.speed_samples.append(r["elapsed_time"])
            if r.get("total_tokens", 0) > 0:
                ms.token_total += r["total_tokens"]
            if r.get("code_verification"):
                ms.code_total += 1
                if r["code_verification"]["success"]:
                    ms.code_passed += 1

    return dict(stats)


# ==================== SVG Charts ====================

class ChartRenderer:
    def __init__(self, colors: List[str], fills: List[str]):
        self.colors = colors
        self.fills = fills

    def radar(self, data: Dict[str, List[float]], categories: List[str], size: int = 440) -> str:
        n = len(categories)
        if n == 0:
            return ""
        center = size // 2
        radius = center - 56
        step = 2 * math.pi / n
        start = -math.pi / 2

        # Smart Dynamic Scaling (Fixed Max=10)
        all_scores = [s for scores in data.values() for s in scores]
        min_score = min(all_scores) if all_scores else 0
        
        axis_max = 10
        proposed_min = max(0, math.floor(min_score) - 2)
        val_range = axis_max - proposed_min

        # Adjust to nice ranges for integer grid labels
        if val_range < 4:
            val_range = 4
        elif val_range == 7:
            val_range = 8
        elif val_range == 9:
            val_range = 10

        axis_min = axis_max - val_range
        
        if val_range in (4, 5):
            grid_steps = val_range
        else: # 6, 8, 10
            grid_steps = val_range // 2
        
        grid, grid_labels = [], []
        for step_i in range(1, grid_steps + 1):
            lvl = step_i / grid_steps
            val = axis_min + step_i * (val_range / grid_steps)
            
            pts = " ".join(
                f"{center + radius * lvl * math.cos(start + i * step):.1f},"
                f"{center + radius * lvl * math.sin(start + i * step):.1f}"
                for i in range(n)
            )
            grid.append(f'<polygon points="{pts}" fill="none" stroke="var(--ink)" stroke-width="0.5" stroke-dasharray="2,3"/>')
            
            gy = center + radius * lvl * math.sin(start)
            display_val = f"{val:.1f}" if val % 1 != 0 else f"{int(val)}"
            grid_labels.append(
                f'<text x="{center + 4}" y="{gy:.1f}" font-size="9" fill="var(--muted)" '
                f'font-family="Libre Baskerville, serif">{display_val}</text>'
            )
            
        # Draw center label
        grid_labels.append(
            f'<text x="{center + 4}" y="{center - 2}" font-size="9" fill="var(--muted)" '
            f'font-family="Libre Baskerville, serif">{axis_min}</text>'
        )

        axes = []
        for i in range(n):
            x = center + radius * math.cos(start + i * step)
            y = center + radius * math.sin(start + i * step)
            axes.append(
                f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" '
                f'stroke="var(--ink)" stroke-width="0.5" stroke-dasharray="2,3"/>'
            )

        labels = []
        for i, cat in enumerate(categories):
            a = start + i * step
            lx = center + (radius + 32) * math.cos(a)
            ly = center + (radius + 32) * math.sin(a)
            anchor = "start" if math.cos(a) > 0.5 else ("end" if math.cos(a) < -0.5 else "middle")
            labels.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" '
                f'fill="var(--muted)" font-family="Libre Baskerville, serif" font-weight="700">{cat}</text>'
            )

        polys, dots, legend = [], [], []
        for idx, (mk, scores) in enumerate(data.items()):
            color = self.colors[idx % len(self.colors)]
            fill_color = self.fills[idx % len(self.fills)]
            pts = []
            for i, s in enumerate(scores):
                a = start + i * step
                # Clamp score strictly into [axis_min, axis_max] for drawing
                clamped_s = max(axis_min, min(axis_max, s))
                r = ((clamped_s - axis_min) / val_range) * radius
                x = center + r * math.cos(a)
                y = center + r * math.sin(a)
                pts.append(f"{x:.1f},{y:.1f}")
                dots.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" '
                    f'stroke="var(--paper)" stroke-width="1" class="radar-dot" data-model="{idx}">'
                    f'<title>{_safe_name(mk)}: {s:.1f}</title></circle>'
                )
            polys.append(
                f'<polygon points="{" ".join(pts)}" fill="{fill_color}" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="miter" class="radar-poly" data-model="{idx}">'
                f'<title>{_safe_name(mk)}</title></polygon>'
            )
            name = _safe_name(mk)
            legend.append(
                f'<span class="chart-legend-item" data-model="{idx}">'
                f'<span class="chart-legend-dot" style="background:{color}"></span>{name}</span>'
            )

        svg = "\n".join(grid + grid_labels + axes + polys + dots + labels)
        return (
            f'<div class="chart-wrap">\n'
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'style="max-width:100%;">{svg}</svg>\n'
            f'<div class="chart-legend">{ "".join(legend) }</div></div>'
        )

    def bar(self, data: Dict[str, List[float]], categories: List[str],
            width: int = 760, height: int = 400, min_bar_width: int = 14) -> str:
        models = list(data.keys())
        n_models = len(models)
        n_cats = len(categories)
        if n_cats == 0 or n_models == 0:
            return ""

        margin = {"top": 30, "right": 20, "bottom": 70, "left": 44}
        cw = width - margin["left"] - margin["right"]
        ch = height - margin["top"] - margin["bottom"]
        gw = cw / n_cats
        bw = gw / (n_models + 1)

        if bw < min_bar_width:
            gw_needed = min_bar_width * (n_models + 1)
            cw = gw_needed * n_cats
            width = cw + margin["left"] + margin["right"]
            gw = cw / n_cats
            bw = gw / (n_models + 1)

        show_labels = bw >= 20
        label_rotate = len(categories) > 6 or any(len(c) > 10 for c in categories)

        bars, x_labels, y_grid = [], [], []
        for i in range(6):
            yv = i * 2
            yp = margin["top"] + ch - (yv / 10) * ch
            y_grid.append(
                f'<line x1="{margin["left"]}" y1="{yp:.1f}" x2="{width - margin["right"]}" '
                f'y2="{yp:.1f}" stroke="var(--ink)" stroke-width="0.5" stroke-dasharray="2,3"/>'
            )
            y_grid.append(
                f'<text x="{margin["left"] - 8}" y="{yp:.1f}" text-anchor="end" '
                f'dominant-baseline="middle" font-size="10" fill="var(--muted)" '
                f'font-family="Libre Baskerville, serif">{yv}</text>'
            )

        for ci, cat in enumerate(categories):
            gx = margin["left"] + ci * gw + gw / 2
            rot = 'transform="rotate(-30,' + f'{gx},{height - margin["bottom"] + 18})"' if label_rotate else ""
            x_labels.append(
                f'<text x="{gx}" y="{height - margin["bottom"] + 18}" text-anchor="{"end" if label_rotate else "middle"}" '
                f'font-size="11" fill="var(--muted)" font-family="Libre Baskerville, serif" font-weight="700" {rot}>{cat}</text>'
            )
            for mi, mk in enumerate(models):
                score = data[mk][ci]
                bx = gx - (n_models * bw) / 2 + mi * bw
                by = margin["top"] + ch - (score / 10) * ch
                bh = (score / 10) * ch
                color = self.colors[mi % len(self.colors)]
                bars.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bw - 2, 2)}" height="{bh:.1f}" '
                    f'fill="{color}" stroke="var(--ink)" stroke-width="1" class="bar-rect" data-model="{mi}">'
                    f'<title>{_safe_name(mk)}: {score:.1f}</title></rect>'
                )
                if show_labels:
                    bars.append(
                        f'<text x="{bx + max(bw - 2, 2) / 2}" y="{by - 4}" text-anchor="middle" '
                        f'font-size="9" fill="var(--muted)" font-family="Libre Baskerville, serif">{score:.1f}</text>'
                    )

        legend = []
        for mi, mk in enumerate(models):
            color = self.colors[mi % len(self.colors)]
            name = _safe_name(mk)
            legend.append(
                f'<span class="chart-legend-item" data-model="{mi}">'
                f'<span class="chart-legend-swatch" style="background:{color}"></span>{name}</span>'
            )

        svg = "\n".join(y_grid + bars + x_labels)
        return (
            f'<div class="chart-wrap" style="overflow-x:auto;">\n'
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'style="min-width:{width}px;">{svg}</svg>\n'
            f'<div class="chart-legend">{ "".join(legend) }</div></div>'
        )

    def heatmap(self, data: Dict[str, Dict[str, float]], categories: List[str], models: List[str]) -> str:
        n_models = len(models)
        font_size = "12px" if n_models <= 8 else "10px"
        header_font = "11px" if n_models <= 8 else "9px"

        cells = []
        for ci, cat in enumerate(categories):
            for mi, mk in enumerate(models):
                score = data.get(mk, {}).get(cat, 0)
                ratio = score / 10.0
                # HSL interpolation: low=red(0,70%,50%), high=green(120,55%,38%)
                h = ratio * 120
                s = 70 - ratio * 15
                l = 52 - ratio * 14
                color = _hsl_to_hex(h, s, l)
                tc = "#fff" if ratio > 0.52 else "var(--ink)"
                cells.append(
                    f'<div class="heatmap-cell" style="background:{color};color:{tc};font-size:{font_size};" '
                    f'title="{_safe_name(mk)} · {CATEGORY_NAMES[cat]}: {score:.1f}">{score:.1f}</div>'
                )

        headers = "".join(
            f'<div class="heatmap-header" style="font-size:{header_font};">{_safe_name(mk)}</div>'
            for mk in models
        )

        rows = ""
        for ci, cat in enumerate(categories):
            rows += f'<div class="heatmap-rowlabel">{CATEGORY_NAMES[cat]}</div>'
            rows += "".join(cells[ci * len(models) + mi] for mi in range(len(models)))

        # Color scale legend
        scale_cells = ""
        for i in range(11):
            val = i
            ratio = val / 10.0
            h = ratio * 120
            s = 70 - ratio * 15
            l = 52 - ratio * 14
            c = _hsl_to_hex(h, s, l)
            scale_cells += f'<div class="heatmap-scale-cell" style="background:{c}"></div>'

        return (
            f'<div class="heatmap-wrap">\n'
            f'<div class="heatmap-grid" style="grid-template-columns: 140px repeat({len(models)}, 1fr);">\n'
            f'<div class="heatmap-corner">Dimension \\ Model</div>\n'
            f'{headers}{rows}</div>\n'
            f'<div class="heatmap-scale">\n'
            f'<span class="heatmap-scale-label" style="font-family:Libre Baskerville,serif;font-size:10px;font-weight:700;">0</span>\n'
            f'{scale_cells}\n'
            f'<span class="heatmap-scale-label" style="font-family:Libre Baskerville,serif;font-size:10px;font-weight:700;">10</span>\n'
            f'</div></div>'
        )


# ==================== CSS ====================

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

:root {
  --paper: #f4f1ea;
  --card: transparent;
  --ink: #111111;
  --ink-light: #333333;
  --muted: #555555;
  --rule: #111111;
  --rule-light: #d4cfc5;
  --accent: #b22222;
  --accent-light: #d65c5c;
  --success: #3A5A3A;
  --danger: #7A2A2A;
  --shadow: rgba(0,0,0,0.05);
  --transition: 0.2s ease;
}

@media (prefers-color-scheme: dark) {
  :root {
    --paper: #1a1a1a;
    --card: transparent;
    --ink: #e6e6e6;
    --ink-light: #cccccc;
    --muted: #999999;
    --rule: #e6e6e6;
    --rule-light: #444444;
    --accent: #d34040;
    --accent-light: #e07070;
    --success: #6A9A6A;
    --danger: #B85A5A;
  }
  .kpi-grid { background: rgba(255,255,255,0.02); }
  .warn-box { background: rgba(255,255,255,0.02); }
  table tr:hover td { background: rgba(255,255,255,0.03); }
  .heatmap-cell { border: 1px solid rgba(0,0,0,0.5); }
}

* { box-sizing: border-box; }

body {
  margin: 0; padding: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: 'EB Garamond', Georgia, serif;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* Subtle paper texture overlay */
body::before {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 9999;
}

.container { max-width: 1024px; margin: 0 auto; padding: 0 40px; }

h1, h2, h3, h4 {
  font-family: 'Playfair Display', Georgia, serif;
  margin: 0; color: var(--ink);
}

/* Masthead - Newspaper Style */
.masthead {
  text-align: center;
  padding: 40px 0 0;
  margin-bottom: 40px;
}
.masthead-title-wrap {
  border-top: 6px solid var(--ink);
  border-bottom: 2px solid var(--ink);
  padding: 24px 0 16px;
  margin-bottom: 8px;
}
.masthead h1 {
  font-size: 80px;
  font-weight: 900;
  letter-spacing: -0.02em;
  line-height: 0.9;
  text-transform: uppercase;
}
.masthead .tagline {
  font-family: 'EB Garamond', serif;
  font-size: 18px;
  font-style: italic;
  color: var(--ink-light);
  margin-top: 12px;
}
.edition-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top: 1px solid var(--ink);
  border-bottom: 4px solid var(--ink);
  padding: 6px 0;
  font-family: 'Libre Baskerville', serif;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
}
.edition-line span { padding: 0 8px; }

/* Typography elements */
h2 {
  font-size: 32px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 8px;
  margin: 48px 0 24px;
}
h3 {
  font-size: 22px;
  font-weight: 700;
  margin: 32px 0 16px;
  font-style: italic;
}

/* Cards -> Sections */
.card, .chart-card {
  margin-bottom: 40px;
}
.chart-title { font-family: 'Playfair Display', serif; font-size: 20px; font-weight: 900; text-transform: uppercase; border-bottom: 1px solid var(--ink); display: inline-block; padding-bottom: 4px; margin-bottom: 8px; }
.chart-subtitle { font-family: 'EB Garamond', serif; font-style: italic; font-size: 14px; color: var(--muted); margin-bottom: 24px; }

/* KPI Box */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-top: 4px solid var(--ink);
  border-bottom: 4px solid var(--ink);
  margin: 32px 0;
  background: rgba(0,0,0,0.02);
}
.kpi-card {
  text-align: center;
  padding: 24px 16px;
  border-right: 1px solid var(--rule-light);
}
.kpi-card:last-child { border-right: none; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 48px; font-weight: 900; color: var(--ink); line-height: 1; margin-bottom: 8px; }
.kpi-label { font-family: 'Libre Baskerville', serif; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.1em; color: var(--accent); }
.kpi-sub { font-family: 'EB Garamond', serif; font-size: 13px; color: var(--muted); font-style: italic; margin-top: 4px; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 24px 0; border-top: 3px solid var(--ink); border-bottom: 3px solid var(--ink); }
th { font-family: 'Libre Baskerville', serif; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--ink); border-bottom: 1px solid var(--ink); padding: 12px 8px; text-align: left; vertical-align: bottom; }
td { padding: 12px 8px; border-bottom: 1px solid var(--rule-light); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(0,0,0,0.03); }
.rank-1 { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 900; color: var(--accent); }
.score-badge { font-family: 'Libre Baskerville', serif; font-size: 14px; font-weight: 700; }

/* Warning */
.warn-box { border: 2px solid var(--ink); padding: 16px; margin: 24px 0; font-style: italic; font-size: 15px; border-left: 6px solid var(--accent); background: rgba(0,0,0,0.02); }
.warn-box strong { font-family: 'Libre Baskerville', serif; font-weight: 700; font-style: normal; text-transform: uppercase; font-size: 12px; color: var(--accent); display: block; margin-bottom: 4px; }
.warn-box code { background: rgba(0,0,0,0.05); padding: 2px 4px; border-radius: 2px; }

/* Detail Cards -> Article layout */
.detail-card { padding: 16px 0; border-bottom: 1px solid var(--rule-light); margin-bottom: 0; }
.detail-card:last-child { border-bottom: none; }
.detail-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.detail-title { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 900; }
.detail-score { font-family: 'Playfair Display', serif; font-size: 24px; font-weight: 900; color: var(--accent); }
.detail-meta { font-family: 'Libre Baskerville', serif; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.detail-reasoning { font-family: 'EB Garamond', serif; font-size: 15px; color: var(--ink-light); line-height: 1.6; padding-left: 16px; border-left: 3px solid var(--rule-light); margin: 12px 0; font-style: italic; background: rgba(0,0,0,0.02); padding-top: 8px; padding-bottom: 8px; }

/* Tags */
.tag { display: inline-block; padding: 2px 6px; font-size: 10px; font-weight: 700; margin-right: 6px; margin-bottom: 4px; text-transform: uppercase; font-family: 'Libre Baskerville', serif; background: var(--ink); color: var(--paper); }
.tag-strength { background: var(--ink); }
.tag-weakness { background: transparent; border: 1px solid var(--ink); color: var(--ink); }

/* Model Groups */
.model-group { border-top: 2px solid var(--ink); padding-top: 16px; margin-bottom: 32px; }
.model-group-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; cursor: pointer; }
.model-group-title { font-family: 'Playfair Display', serif; font-size: 24px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.02em; }
.model-group-toggle { font-family: 'Libre Baskerville', serif; font-size: 14px; font-weight: 700; transition: transform var(--transition); }
.model-group.collapsed .model-group-toggle { transform: rotate(-90deg); }
.model-group.collapsed .model-group-body { display: none; }
.model-group.collapsed .model-group-header { margin-bottom: 0; }

/* Stats Grid for Summaries */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1px; background: var(--rule-light); border: 1px solid var(--rule-light); margin: 16px 0; }
.stat-box { background: var(--paper); padding: 16px 12px; text-align: center; }
.stat-value { font-family: 'Playfair Display', serif; font-size: 24px; font-weight: 900; line-height: 1; margin-bottom: 6px; color: var(--ink); }
.stat-label { font-family: 'Libre Baskerville', serif; font-size: 10px; color: var(--muted); text-transform: uppercase; font-weight: 700; }

/* Progress tracks */
.progress-track { width: 100%; height: 4px; background: var(--rule-light); }
.progress-fill { height: 100%; background: var(--ink); }

/* Compare Grid */
.compare-grid { display: grid; grid-template-columns: 140px repeat(auto-fit, minmax(100px, 1fr)); gap: 1px; background: var(--ink); border: 2px solid var(--ink); margin: 24px 0; }
.compare-cell { background: var(--paper); padding: 12px; font-size: 14px; text-align: center; }
.compare-cell.header { font-family: 'Libre Baskerville', serif; font-weight: 700; font-size: 11px; text-transform: uppercase; background: rgba(0,0,0,0.05); }
.compare-cell.model { font-family: 'Playfair Display', serif; font-weight: 900; text-align: left; background: rgba(0,0,0,0.02); }

/* Charts Layout */
.chart-wrap { margin: 24px 0; display: flex; flex-direction: column; align-items: center; }
.chart-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin-top: 16px; border-top: 1px solid var(--rule-light); padding-top: 16px; width: 100%; }
.chart-legend-item { font-family: 'Libre Baskerville', serif; font-size: 11px; font-weight: 700; text-transform: uppercase; cursor: pointer; display: flex; align-items: center; opacity: 0.8; }
.chart-legend-item:hover { opacity: 1; }
.chart-legend-item.inactive { opacity: 0.3; }
.chart-legend-swatch { width: 12px; height: 12px; margin-right: 6px; border: 1px solid var(--ink); }
.chart-legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; border: 1px solid var(--ink); }

/* Radar interactions */
.radar-poly { transition: opacity var(--transition), stroke-width var(--transition), fill-opacity var(--transition); fill-opacity: 0.6; mix-blend-mode: multiply; }
.radar-poly.dimmed { opacity: 0.05; }
.radar-poly.highlighted { stroke-width: 3.5; opacity: 1; fill-opacity: 0.9; }
.radar-dot { transition: opacity var(--transition), r var(--transition); }
.radar-dot.dimmed { opacity: 0; }
.radar-dot.highlighted { r: 5; stroke-width: 1.5; }

@media (prefers-color-scheme: dark) {
  .radar-poly { mix-blend-mode: screen; }
}

/* Bar interactions */
.bar-rect { transition: opacity var(--transition); }
.bar-rect.dimmed { opacity: 0.15; }
.bar-rect.highlighted { opacity: 1; filter: brightness(1.1); }

/* Heatmap */
.heatmap-wrap { overflow-x: auto; margin: 24px 0; border: 2px solid var(--ink); padding: 2px; }
.heatmap-grid { display: grid; gap: 1px; background: var(--ink); }
.heatmap-corner, .heatmap-header, .heatmap-rowlabel, .heatmap-cell { background: var(--paper); padding: 8px; }
.heatmap-corner { font-family: 'Libre Baskerville', serif; font-size: 10px; font-weight: 700; text-transform: uppercase; text-align: right; }
.heatmap-header { font-family: 'Playfair Display', serif; font-size: 12px; font-weight: 900; text-align: center; }
.heatmap-rowlabel { font-family: 'Libre Baskerville', serif; font-size: 11px; font-weight: 700; text-transform: uppercase; text-align: right; }
.heatmap-cell { font-family: 'Libre Baskerville', serif; text-align: center; font-weight: 700; transition: filter 0.2s; border: 1px solid rgba(0,0,0,0.05); }
.heatmap-cell:hover { filter: brightness(1.1) saturate(1.2); }
.heatmap-scale { display: flex; justify-content: flex-end; padding: 8px 0 0; }
.heatmap-scale-cell { width: 24px; height: 8px; }

/* Buttons */
.action-bar { border-bottom: 2px solid var(--ink); padding-bottom: 16px; margin-bottom: 32px; text-align: right; }
.btn { display: inline-flex; align-items: center; gap: 8px; font-family: 'Libre Baskerville', serif; font-size: 11px; font-weight: 700; text-transform: uppercase; background: var(--ink); color: var(--paper); border: none; padding: 8px 16px; cursor: pointer; transition: background 0.2s; border-radius: 0; }
.btn:hover { background: var(--accent); color: var(--paper); }
.btn svg { width: 14px; height: 14px; }

.scroll-h { overflow-x: auto; }

/* Footer */
.report-footer { border-top: 4px solid var(--ink); text-align: center; padding: 40px 0; margin-top: 64px; font-family: 'Libre Baskerville', serif; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink); }
.footer-name { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 900; margin-bottom: 12px; letter-spacing: 0; text-transform: none; }

@media(max-width:768px){
  .container{padding:0 20px}
  .masthead h1{font-size:48px}
  h2{font-size:22px}
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .kpi-card:nth-child(2n){border-right:none}
  .kpi-card{border-bottom:1px solid var(--rule-light)}
}

@media print {
  body::before {display:none}
  .card, .chart-card, .detail-card {box-shadow:none; break-inside:avoid}
  .model-group {break-inside:avoid}
  .action-bar {display:none}
  .heatmap-wrap {overflow:visible}
}
"""

# ==================== JavaScript ====================

_JS = """
document.addEventListener('DOMContentLoaded', function() {
  // Toggle model group details
  document.querySelectorAll('.model-group-header').forEach(function(header) {
    header.addEventListener('click', function() {
      var group = this.closest('.model-group');
      group.classList.toggle('collapsed');
    });
  });

  // Highlight model on legend hover
  var legends = document.querySelectorAll('.chart-legend-item');
  legends.forEach(function(item) {
    item.addEventListener('mouseenter', function() {
      var idx = this.getAttribute('data-model');
      if (idx === null) return;
      document.querySelectorAll('.radar-poly, .bar-rect, .radar-dot').forEach(function(el) {
        if (el.getAttribute('data-model') === idx) {
          el.classList.add('highlighted');
          el.classList.remove('dimmed');
        } else {
          el.classList.add('dimmed');
          el.classList.remove('highlighted');
        }
      });
    });
    item.addEventListener('mouseleave', function() {
      document.querySelectorAll('.radar-poly, .bar-rect, .radar-dot').forEach(function(el) {
        el.classList.remove('highlighted', 'dimmed');
      });
    });
    item.addEventListener('click', function() {
      this.classList.toggle('inactive');
    });
  });

  // CSV export
  var exportBtn = document.getElementById('btn-export');
  if (exportBtn) {
    exportBtn.addEventListener('click', function() {
      var table = document.getElementById('leaderboard-table');
      if (!table) return;
      var rows = table.querySelectorAll('tr');
      var csv = [];
      rows.forEach(function(row) {
        var cols = row.querySelectorAll('th, td');
        var line = [];
        cols.forEach(function(col) {
          var text = col.textContent.replace(/"/g, '""').trim();
          line.push('"' + text + '"');
        });
        csv.push(line.join(','));
      });
      var blob = new Blob([csv.join('\\n')], {type: 'text/csv;charset=utf-8;'});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'veilbench_leaderboard.csv';
      a.click();
      URL.revokeObjectURL(url);
    });
  }
});
"""

# ==================== Report Builder ====================

class HTMLBuilder:
    """Helper for building HTML strings efficiently."""

    def __init__(self):
        self.parts: List[str] = []

    def add(self, *lines: str) -> "HTMLBuilder":
        self.parts.extend(lines)
        return self

    def tag(self, name: str, content: str, **attrs) -> "HTMLBuilder":
        attr_str = "".join(f' {k}="{v}"' for k, v in attrs.items())
        self.parts.append(f"<{name}{attr_str}>{content}</{name}>")
        return self

    def raw(self, html: str) -> "HTMLBuilder":
        self.parts.append(html)
        return self

    def build(self) -> str:
        return "\n".join(self.parts)


class ReportGenerator:
    def __init__(self, stats: Dict[str, ModelStats], colors: List[str], fills: List[str], is_obj_only: bool):
        self.stats = stats
        self.colors = colors
        self.fills = fills
        self.is_obj_only = is_obj_only
        self.charts = ChartRenderer(colors, fills)
        self.models = sorted(stats.keys(), key=lambda mk: stats[mk].avg_overall, reverse=True)
        self.n_models = len(self.models)

    def _model_name(self, mk: str) -> str:
        return _safe_name(mk)

    def generate(self, output_path: str) -> None:
        hb = HTMLBuilder()

        # Head
        hb.add(
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>VeilBench — Evaluation Report</title>',
            f'<style>{_CSS}</style>',
            '</head>',
            '<body>',
        )

        # Masthead
        subtitle = "Thinking Mode Benchmark"
        if self.is_obj_only:
            subtitle += " &mdash; Objective Only"

        hb.add(
            '<header class="masthead">',
            '<div class="container">',
            '<div class="masthead-title-wrap">',
            '<h1>VeilBench</h1>',
            f'<p class="tagline">{subtitle}</p>',
            '</div>',
            '<div class="edition-line">',
            f'<span>Vol. 1 &mdash; {datetime.now().strftime("%B %d, %Y")}</span>',
            '<span>' + ' &middot; '.join([
                f'{self.n_models} Models',
                f'{len(ALL_TESTS)} Problems',
                f'{len(CATEGORY_ORDER)} Dimensions'
            ]) + '</span>',
            '<span>THE BENCHMARK</span>',
            '</div>',
            '</div>',
            '</header>',
        )

        hb.add('<main class="container">')

        hb.add(
            '<div style="column-count: 2; column-gap: 40px; margin-bottom: 32px; font-size: 15px; text-align: justify; border-bottom: 2px solid var(--ink); padding-bottom: 32px;">',
            '<p style="margin-top:0;"><span style="float:left; font-family:\'Playfair Display\',serif; font-size:64px; line-height:0.8; padding-right:8px; padding-top:4px; font-weight:900;">T</span>his report provides an exhaustive multi-dimensional evaluation of reasoning models, testing their limits across logic, math, system design, and algorithmic tasks. The scoring methodology blends automated objective verification with blinded LLM-as-a-judge subjective evaluation.</p>',
            '<p>The following sections present a comprehensive leaderboard, performance heatmaps, and granular breakdowns of each model\'s specific strengths and weaknesses. Evaluation integrity is maintained through fully anonymized prompts and reproducible sandboxed environments.</p>',
            '</div>'
        )

        # Warning box
        if self.is_obj_only:
            hb.add(
                '<div class="warn-box">',
                '<strong>Objective-Only Mode.</strong> This report is based solely on objective validation scores. ',
                'Subjective evaluation has not been performed. Run <code>python3 evaluate.py</code> for the full report.',
                '</div>'
            )

        # KPI
        best_score = self.stats[self.models[0]].avg_overall if self.models else 0
        all_speeds = [s for ms in self.stats.values() for s in ms.speed_samples]
        avg_speed = _avg(all_speeds)
        total_code = sum(ms.code_total for ms in self.stats.values())
        code_rate = (sum(ms.code_passed for ms in self.stats.values()) / total_code * 100) if total_code else 0
        total_tok = sum(ms.token_total for ms in self.stats.values())
        avg_median = _avg([ms.median_overall for ms in self.stats.values()])

        hb.add('<div class="kpi-grid">')
        hb.add(
            f'<div class="kpi-card"><div class="kpi-value">{best_score:.2f}</div><div class="kpi-label">Highest Average</div></div>',
            f'<div class="kpi-card"><div class="kpi-value">{avg_speed:.1f}s</div><div class="kpi-label">Avg Response</div><div class="kpi-sub">median {_median(all_speeds):.1f}s</div></div>',
            f'<div class="kpi-card"><div class="kpi-value">{code_rate:.0f}%</div><div class="kpi-label">Code Pass Rate</div><div class="kpi-sub">{total_code} tests</div></div>',
            f'<div class="kpi-card"><div class="kpi-value">{total_tok/1000:.0f}k</div><div class="kpi-label">Total Tokens</div><div class="kpi-sub">avg {avg_median:.2f} median</div></div>',
        )
        hb.add('</div>')

        # Action bar
        hb.add(
            '<div class="action-bar">',
            '<button class="btn" id="btn-export">',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
            'Export CSV',
            '</button>',
            '</div>'
        )

        # Leaderboard
        hb.add('<h2>Leaderboard</h2>')
        hb.add('<div class="card scroll-h">')
        hb.add('<table id="leaderboard-table">')
        hb.add('<thead><tr><th>Rank</th><th>Model</th><th>Overall</th><th>Median</th><th>σ</th>')
        for c in CATEGORY_ORDER:
            hb.add(f'<th>{CATEGORY_NAMES[c]}</th>')
        hb.add('<th>Speed</th><th>Code</th></tr></thead>')
        hb.add('<tbody>')
        for rank, mk in enumerate(self.models, 1):
            ms = self.stats[mk]
            rc = f"rank-{rank}" if rank <= 3 else ""
            cat_vals = "".join(f'<td>{ms.cat_avg(c):.1f}</td>' for c in CATEGORY_ORDER)
            code_str = f"{ms.code_passed}/{ms.code_total}" if ms.code_total > 0 else "&mdash;"
            hb.add(
                f'<tr>',
                f'<td class="{rc}">#{rank}</td>',
                f'<td style="font-weight:600;">{self._model_name(mk)}</td>',
                f'<td><span class="score-badge">{ms.avg_overall:.2f}</span></td>',
                f'<td>{ms.median_overall:.2f}</td>',
                f'<td>{ms.std_overall:.2f}</td>',
                cat_vals,
                f'<td>{ms.avg_speed:.1f}s</td>',
                f'<td>{code_str}</td>',
                f'</tr>'
            )
        hb.add('</tbody></table></div>')

        # Charts
        radar_data = {mk: [self.stats[mk].cat_avg(c) for c in CATEGORY_ORDER] for mk in self.models}
        heat_data = {mk: {c: round(self.stats[mk].cat_avg(c), 1) for c in CATEGORY_ORDER} for mk in self.models}

        hb.add('<div class="chart-card">')
        hb.add('<div class="chart-title">Capability Radar</div>')
        hb.add('<div class="chart-subtitle">Multi-dimensional performance comparison (0–10 scale)</div>')
        hb.raw(self.charts.radar(radar_data, [CATEGORY_NAMES[c] for c in CATEGORY_ORDER], size=440))
        hb.add('</div>')

        hb.add('<div class="chart-card">')
        hb.add('<div class="chart-title">Dimension Breakdown</div>')
        hb.add('<div class="chart-subtitle">Average scores per category</div>')
        hb.raw(self.charts.bar(radar_data, [CATEGORY_NAMES[c] for c in CATEGORY_ORDER], width=760, height=400))
        hb.add('</div>')

        if not self.is_obj_only:
            hb.add('<div class="chart-card">')
            hb.add('<div class="chart-title">Objective vs Subjective</div>')
            hb.add('<div class="chart-subtitle">Objective 40% vs Subjective 60% weighting</div>')
            obj_subj = {}
            for mk in self.models:
                ms = self.stats[mk]
                obj_val = _avg([s for v in ms.obj_scores.values() for s in v])
                subj_val = _avg([s for v in ms.subj_scores.values() for s in v])
                obj_subj[mk] = [obj_val, subj_val]
            hb.raw(self.charts.bar(obj_subj, ["Objective", "Subjective"], width=520, height=300))
            hb.add('</div>')

        hb.add('<div class="chart-card">')
        hb.add('<div class="chart-title">Performance Heatmap</div>')
        hb.add('<div class="chart-subtitle">Score strength per model and dimension</div>')
        hb.raw(self.charts.heatmap(heat_data, CATEGORY_ORDER, self.models))
        hb.add('</div>')

        # Comparison Matrix
        if self.n_models >= 2:
            hb.add('<h2>Head-to-Head Comparison</h2>')
            hb.add('<div class="card scroll-h">')
            hb.add('<div class="compare-grid">')
            hb.add('<div class="compare-cell header">Model</div>')
            for mk in self.models:
                hb.add(f'<div class="compare-cell header">{self._model_name(mk)}</div>')
            for mk in self.models:
                hb.add(f'<div class="compare-cell model">{self._model_name(mk)}</div>')
                for mk2 in self.models:
                    if mk == mk2:
                        hb.add('<div class="compare-cell" style="background:rgba(139,26,26,0.04)">—</div>')
                    else:
                        diff = self.stats[mk].avg_overall - self.stats[mk2].avg_overall
                        sign = "+" if diff > 0 else ""
                        color = "var(--success)" if diff > 0 else ("var(--danger)" if diff < 0 else "var(--muted)")
                        hb.add(f'<div class="compare-cell" style="color:{color};font-weight:700">{sign}{diff:.2f}</div>')
            hb.add('</div></div>')

        # Detailed Analysis
        hb.add('<h2>Detailed Analysis</h2>')
        for cat in CATEGORY_ORDER:
            cat_name = CATEGORY_NAMES[cat]
            hb.add(f'<h3>{cat_name}</h3>')
            test_map = defaultdict(list)
            for mk in self.models:
                for d in self.stats[mk].details:
                    if d.category == cat:
                        test_map[d.test_id].append((mk, d))
            for tid in sorted(test_map.keys()):
                items = test_map[tid]
                test_name = items[0][1].test_name
                hb.add(f'<div class="model-group">')
                hb.add(
                    f'<div class="model-group-header">',
                    f'<span class="model-group-title">{test_name}</span>',
                    f'<span class="model-group-toggle">▼</span>',
                    f'<span style="font-size:12px;color:var(--muted);font-family:EB Garamond,serif;">{tid}</span>',
                    f'</div>'
                )
                hb.add('<div class="model-group-body">')
                for mk, d in items:
                    tags = "".join(f'<span class="tag tag-strength">{s}</span>' for s in (d.strengths or [])[:2])
                    tags += "".join(f'<span class="tag tag-weakness">{w}</span>' for w in (d.weaknesses or [])[:2])
                    code_icon = "&#10003;" if d.code_passed else ("&#10007;" if d.code_passed is not None else "")
                    code_style = "color:var(--success)" if d.code_passed else ("color:var(--danger)" if d.code_passed is not None else "")
                    obj_tag = f' &middot; Obj: {d.objective_score:.1f}' if d.objective_score is not None else ''
                    reasoning_preview = d.reasoning[:MAX_REASONING_PREVIEW]
                    if len(d.reasoning) > MAX_REASONING_PREVIEW:
                        reasoning_preview += "…"
                    # Escape HTML in reasoning
                    reasoning_preview = reasoning_preview.replace("<", "&lt;").replace(">", "&gt;")

                    hb.add(f'<div class="detail-card">')
                    hb.add(f'<div class="detail-header">')
                    hb.add(f'<div>')
                    hb.add(f'<div class="detail-title">{self._model_name(mk)}</div>')
                    hb.add(f'<div class="detail-meta">{d.test_id} <span style="{code_style}">{code_icon}</span></div>')
                    hb.add(f'</div>')
                    hb.add(f'<div class="detail-score">{d.overall:.1f}</div>')
                    hb.add(f'</div>')
                    if tags:
                        hb.add(f'<div style="margin-bottom:10px;">{tags}</div>')
                    hb.add(f'<div class="progress-track"><div class="progress-fill" style="width:{d.overall*10:.1f}%"></div></div>')
                    if reasoning_preview:
                        hb.add(f'<div class="detail-reasoning">{reasoning_preview}</div>')
                    hb.add(
                        f'<div style="font-size:11px;color:var(--muted);margin-top:10px;text-align:right;font-family:EB Garamond,serif;">',
                        f'{d.elapsed_time:.1f}s &middot; {d.completion_tokens} tokens{obj_tag}',
                        f'</div>'
                    )
                    hb.add(f'</div>')
                hb.add('</div></div>')

        # Model Summaries
        hb.add('<h2>Model Summaries</h2>')
        for mk in self.models:
            ms = self.stats[mk]
            best_cat, best_val = ms.best_cat()
            worst_cat, worst_val = ms.worst_cat()
            obj_avg = _avg([s for v in ms.obj_scores.values() for s in v])
            subj_avg = _avg([s for v in ms.subj_scores.values() for s in v])

            hb.add(f'<div class="card">')
            hb.add(f'<div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:12px;">')
            hb.add(f'<h3 style="margin:0;">{self._model_name(mk)}</h3>')
            hb.add(f'<span class="score-badge">{ms.avg_overall:.2f}</span>')
            hb.add(f'</div>')

            # Stats grid
            hb.add(f'<div class="stats-grid">')
            hb.add(
                f'<div class="stat-box">',
                f'<div class="stat-value">{ms.median_overall:.2f}</div>',
                f'<div class="stat-label">Median</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{ms.std_overall:.2f}</div>',
                f'<div class="stat-label">Std Dev</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{CATEGORY_NAMES.get(best_cat, best_cat)}</div>',
                f'<div class="stat-label">Best · {best_val:.1f}</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{CATEGORY_NAMES.get(worst_cat, worst_cat)}</div>',
                f'<div class="stat-label">Weakest · {worst_val:.1f}</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{obj_avg:.2f}</div>',
                f'<div class="stat-label">Objective</div>',
                f'</div>',
            )
            if not self.is_obj_only:
                hb.add(
                    f'<div class="stat-box">',
                    f'<div class="stat-value">{subj_avg:.2f}</div>',
                    f'<div class="stat-label">Subjective</div>',
                    f'</div>',
                )
            hb.add(
                f'<div class="stat-box">',
                f'<div class="stat-value">{ms.avg_speed:.1f}s</div>',
                f'<div class="stat-label">Speed</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{ms.token_total/1000:.0f}k</div>',
                f'<div class="stat-label">Tokens</div>',
                f'</div>',
                f'<div class="stat-box">',
                f'<div class="stat-value">{ms.code_passed}/{ms.code_total}</div>',
                f'<div class="stat-label">Code</div>',
                f'</div>',
            )
            hb.add(f'</div>')

            # Dimension bars
            hb.add(f'<div style="margin-top:16px;">')
            for c in CATEGORY_ORDER:
                val = ms.cat_avg(c)
                hb.add(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">',
                    f'<div style="width:100px;font-size:11px;color:var(--muted);text-align:right;font-family:EB Garamond,serif;">{CATEGORY_NAMES[c]}</div>',
                    f'<div class="progress-track" style="flex:1;height:6px;">',
                    f'<div class="progress-fill" style="width:{val*10:.1f}%"></div>',
                    f'</div>',
                    f'<div style="width:36px;font-size:11px;font-weight:700;font-family:Libre Baskerville,serif;">{val:.1f}</div>',
                    f'</div>'
                )
            hb.add(f'</div></div>')

        hb.add('</main>')

        # Footer
        hb.add(
            f'<footer class="report-footer">',
            f'<div class="footer-name">VeilBench</div>',
            f'<div>Thinking Mode Benchmark &middot; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            f'</footer>'
        )

        # Scripts
        hb.add(f'<script>{_JS}</script>')
        hb.add('</body></html>')

        # Write
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(hb.build())
        print(f"Report generated: {output_path}")


# ==================== CLI ====================

def generate_report(output_path: Optional[str] = None,
                    selected_models: Optional[List[str]] = None,
                    objective_only: bool = False) -> None:
    output_path = output_path or DEFAULT_REPORT_PATH

    if not os.path.isdir(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        sys.exit(1)

    has_evaluated = os.path.exists(os.path.join(RESULTS_DIR, "evaluated_results.json"))

    if objective_only or not has_evaluated:
        raw = load_objective_only_data()
        stats = build_from_objective(raw, selected_models)
        is_obj_only = True
    else:
        results = load_evaluated_data()
        stats = build_from_evaluated(results, selected_models)
        is_obj_only = False

    if not stats:
        print("Error: No data available for the selected models.")
        return

    colors, fills = generate_palette(len(stats))
    gen = ReportGenerator(stats, colors, fills, is_obj_only)
    gen.generate(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="VeilBench Report Generator")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys to include (e.g. m1,m2). Default: all.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path. Default: report.html")
    parser.add_argument("--objective-only", action="store_true",
                        help="Generate report using only objective validation scores.")
    args = parser.parse_args()

    selected = None
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        invalid = [m for m in selected if m not in MODELS]
        if invalid:
            print(f"Error: Unknown model keys: {invalid}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)

    generate_report(output_path=args.output, selected_models=selected, objective_only=args.objective_only)


if __name__ == "__main__":
    main()
