"""
VeilBench — 客观验证器集合（增强版）
数学、文本、代码等多类型自动验证
改进点：
1. math_range 支持上下文感知、分数解析、百分比、场景区分
2. exact_text 支持正则与语义扩展匹配
3. 引入 expected_answer 辅助验证
4. python_exec 增强代码提取与错误诊断
"""

import re
import json
import math
import tempfile
import subprocess
import os
from typing import Dict, Any, Tuple, List, Optional


# ==================== 数值智能提取 ====================

def _parse_number(s: str) -> Optional[float]:
    """解析数字字符串，支持分数"""
    s = s.strip()
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_context(text: str, start: int, end: int, window: int = 60) -> str:
    """提取匹配位置附近的上下文"""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end]


def _smart_extract_numbers(answer: str) -> List[Tuple[float, str]]:
    """
    智能提取数值，返回 [(value, context), ...]
    支持：分数（包括LaTeX \frac）、百分比、键值对、普通数字
    """
    numbers = []
    seen_positions = set()

    # 1. 提取 LaTeX 分数：\frac{a}{b}
    for m in re.finditer(r'\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}', answer):
        val = _parse_number(f"{m.group(1)}/{m.group(2)}")
        if val is not None:
            context = _extract_context(answer, m.start(), m.end(), window=120)
            numbers.append((val, context))
            seen_positions.add((m.start(), m.end()))

    # 2. 提取普通分数形式：243/275
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\b', answer):
        pos = (m.start(), m.end())
        if pos in seen_positions:
            continue
        val = _parse_number(f"{m.group(1)}/{m.group(2)}")
        if val is not None:
            context = _extract_context(answer, m.start(), m.end(), window=120)
            numbers.append((val, context))
            seen_positions.add(pos)

    # 2. 提取键值对：key = value / key ≈ value / key: value / 中文键：值
    kv_pattern = re.compile(
        r'([A-Za-z_\u4e00-\u9fa5][A-Za-z0-9_\u4e00-\u9fa5\s\(\)\[\]\{\}]{0,40}?)\s*[=≈:：]\s*([\d\.]+)\s*(%?)',
        re.UNICODE
    )
    for m in kv_pattern.finditer(answer):
        key = m.group(1).strip()
        val = _parse_number(m.group(2))
        if val is None:
            continue
        is_percent = bool(m.group(3)) or '%' in m.group(0)
        if is_percent:
            val = val / 100.0
        numbers.append((val, key))
        seen_positions.add((m.start(), m.end()))

    # 3. 提取带百分号的数字
    for m in re.finditer(r'([\d\.]+)\s*%', answer):
        val = _parse_number(m.group(1))
        if val is not None:
            val = val / 100.0
            context = _extract_context(answer, m.start(), m.end(), window=120)
            numbers.append((val, context))
            seen_positions.add((m.start(), m.end()))

    # 4. 提取普通小数（避免与已匹配的分数/百分比重叠）
    for m in re.finditer(r'\b(\d+\.\d+)\b', answer):
        pos = (m.start(), m.end())
        if pos in seen_positions:
            continue
        val = _parse_number(m.group(1))
        if val is not None:
            context = _extract_context(answer, m.start(), m.end(), window=120)
            numbers.append((val, context))

    # 5. 提取普通整数（过滤明显是编号的，如 2024、年份等）
    for m in re.finditer(r'\b(\d+)\b', answer):
        pos = (m.start(), m.end())
        if pos in seen_positions:
            continue
        val = _parse_number(m.group(1))
        if val is not None:
            context = _extract_context(answer, m.start(), m.end(), window=120)
            numbers.append((val, context))

    return numbers


def _context_matches(context: str, key: str) -> float:
    """
    判断上下文是否与 key 相关。
    返回 0~1 的相关度分数。
    """
    key_lower = key.lower()
    context_lower = context.lower()

    # 构建 key 的语义映射
    semantic_map = {
        'prob': ['概率', 'p(', 'prob', '吸收', '被位置', '被10', 'u_5', 'u_10'],
        'expected_steps': ['期望', '停时', '期望步数', '步数', 'v_i', '平均步数', 'τ'],
        'net_gain': ['净收益', '收益', '期望净', '利润', 'gain', '净'],
        'p1': ['p1', 'p_1', '价格1', '卖家1'],
        'p2': ['p2', 'p_2', '价格2', '卖家2'],
        'p3': ['p3', 'p_3', '价格3', '卖家3'],
        'p4': ['p4', 'p_4', '价格4', '卖家4'],
        'p5': ['p5', 'p_5', '价格5', '卖家5'],
        'profit': ['利润', '收益', 'profit', 'π'],
        'no_signal': ['无信号', 'no signal', '无信号时', '均衡'],
        'p1_normal': ['第一次', 'first', 'p1', '后验概率'],
        'p2_normal': ['第二次', '两次', 'p2'],
        'p3_normal': ['第三次', '阴性', 'p3'],
        'p1_high': ['高风险', 'high', 'p1'],
        'p2_high': ['高风险', 'high', 'p2'],
        'p3_high': ['高风险', 'high', 'p3'],
        'success_rate': ['成功率', '概率', 'success', '约', '≈', 'P(', 'percent', '%'],
        'exact_weight': ['权重', '精确', '最小', 'exact', 'weight', '斯坦纳'],
        'heuristic_weight': ['启发式', '近似', 'heuristic', '近似比', '最短路径'],
        'channel_capacity': ['信道容量', 'capacity', '信道', '敲击'],
    }

    # 精确匹配 key 中的子串
    for sem_key, hints in semantic_map.items():
        if sem_key in key_lower:
            for hint in hints:
                if hint.lower() in context_lower:
                    return 1.0

    # 如果上下文中包含 key 本身的子串
    if key_lower in context_lower:
        return 0.8

    # 数字 proximity：如果上下文中有明显的数学表达式标记
    math_markers = ['=', '≈', '≃', '→', '：', ':', '为', '是', '约']
    if any(m in context for m in math_markers):
        return 0.3

    return 0.0


def _scenario_filter(numbers: List[Tuple[float, str]], scenario_hint: str = None) -> List[Tuple[float, str]]:
    """
    根据场景提示过滤数值列表。
    例如 scenario_hint='06' 表示找 p=0.6 场景下的数值，
    scenario_hint='05' 表示 p=0.5 场景。
    如果过滤后结果为空，回退到全部数值。
    """
    if not scenario_hint:
        return numbers

    filtered = []
    for val, ctx in numbers:
        # 检查上下文是否包含场景标识
        if scenario_hint in ctx:
            filtered.append((val, ctx))
            continue
        # 模糊匹配
        if scenario_hint == '06' and ('0.6' in ctx or 'p=0.6' in ctx or 'p = 0.6' in ctx):
            filtered.append((val, ctx))
            continue
        if scenario_hint == '05' and ('0.5' in ctx or 'p=0.5' in ctx or 'p = 0.5' in ctx):
            filtered.append((val, ctx))
            continue
        if scenario_hint == 'normal' and ('普通' in ctx or '一般' in ctx or '0.1%' in ctx or '患病率' in ctx):
            filtered.append((val, ctx))
            continue
        if scenario_hint == 'high' and ('高风险' in ctx or '5%' in ctx or '风险' in ctx):
            filtered.append((val, ctx))
            continue

    # 关键改进：如果场景过滤后为空，回退到全部数值
    return filtered if filtered else numbers


# ==================== 核心验证器 ====================

def validate_math_range(answer: str, config: Dict[str, Any]) -> Tuple[float, str]:
    """
    从答案中智能提取数值，验证是否在指定范围内。
    改进：上下文感知、场景区分、分数解析、百分比正确识别。
    新增：支持 must_contain / must_not_contain 文本语义检查。
    """
    tolerance = config.get("tolerance", 0.1)
    key_values = config.get("key_values", {})
    must_contain = config.get("must_contain", [])
    must_not_contain = config.get("must_not_contain", [])

    # 先进行文本语义检查
    text_score = 1.0
    text_details = []
    for word in must_contain:
        found = False
        if word in answer:
            found = True
        else:
            try:
                if re.search(word, answer):
                    found = True
            except re.error:
                pass
        if not found:
            # 语义扩展回退
            expanded = _semantic_expand(word)
            for syn in expanded:
                if syn in answer:
                    found = True
                    break
        if found:
            text_details.append(f"✓ 包含必要文本: '{word}'")
        else:
            text_details.append(f"✗ 缺少必要文本: '{word}'")
            text_score -= 1.0 / max(len(must_contain), 1)

    for word in must_not_contain:
        # 使用正则进行更精确的匹配，要求禁用词作为独立语义单元出现
        # 支持用户传入正则（如 r'不存在.*策略'），也支持普通词的边界匹配
        found = False
        try:
            # 尝试作为正则匹配
            if re.search(word, answer):
                found = True
        except re.error:
            # 普通词：要求前后不是字母/数字/中文，避免子串误匹配
            pattern = r'(?<![A-Za-z0-9\u4e00-\u9fa5])' + re.escape(word) + r'(?![A-Za-z0-9\u4e00-\u9fa5])'
            if re.search(pattern, answer):
                found = True
        if found:
            text_details.append(f"✗ 包含禁用文本: '{word}'")
            text_score = 0.0
        else:
            text_details.append(f"✓ 未包含禁用文本: '{word}'")

    # 如果没有配置数值检查，直接返回文本分数
    if not key_values:
        return max(0.0, text_score), "\n".join(text_details)

    # 提取所有数值及其上下文
    all_numbers = _smart_extract_numbers(answer)

    scores = []
    details = []
    details.extend(text_details)

    for key, (low, high) in key_values.items():
        # 推断场景提示
        scenario_hint = None
        if '_06' in key or 'prob_10_06' == key or 'expected_steps_06' == key or 'net_gain_06' == key:
            scenario_hint = '06'
        elif '_05' in key or 'prob_10_05' == key or 'expected_steps_05' == key or 'net_gain_05' == key:
            scenario_hint = '05'
        elif '_normal' in key:
            scenario_hint = 'normal'
        elif '_high' in key:
            scenario_hint = 'high'

        # 先尝试场景过滤
        candidate_numbers = _scenario_filter(all_numbers, scenario_hint)

        # 优先尝试上下文匹配
        found = False
        best_match = None
        best_score = 0.0

        for val, ctx in candidate_numbers:
            # 检查数值是否在范围内（含容忍度）
            in_range = low - tolerance <= val <= high + tolerance
            if not in_range:
                continue

            # 计算上下文相关度
            relevance = _context_matches(ctx, key)
            if relevance >= 0.8:
                found = True
                best_match = val
                break
            elif relevance >= 0.3 and relevance > best_score:
                best_score = relevance
                best_match = val

        # 如果上下文匹配失败，但有高置信度数值在范围内，也接受
        if not found and best_match is not None:
            found = True
            best_match = best_match

        # 回退1：在场景过滤结果中全局搜索
        if not found:
            for val, ctx in candidate_numbers:
                if low - tolerance <= val <= high + tolerance:
                    found = True
                    best_match = val
                    break

        # 回退2：在全部数值中搜索（场景过滤可能误删了正确数值）
        if not found:
            for val, ctx in all_numbers:
                if low - tolerance <= val <= high + tolerance:
                    found = True
                    best_match = val
                    break

        if found:
            scores.append(1.0)
            details.append(f"✓ {key}: 找到符合范围的数值 [{low}, {high}] (匹配值≈{best_match})")
        else:
            scores.append(0.0)
            details.append(f"✗ {key}: 未找到 [{low}, {high}] 范围内的数值")

    # 数值分数与文本分数加权融合
    num_score = sum(scores) / len(scores) if scores else 0.0
    if must_contain or must_not_contain:
        # 数值占 70%，文本占 30%
        final_score = 0.7 * num_score + 0.3 * max(0.0, text_score)
    else:
        final_score = num_score
    return final_score, "\n".join(details)


def validate_exact_text(answer: str, config: Dict[str, Any]) -> Tuple[float, str]:
    """
    验证答案是否包含必须关键词，且不包含禁用词。
    改进：支持正则表达式匹配；增加语义扩展。
    """
    must_contain = config.get("must_contain", [])
    forbidden = config.get("forbidden_words", [])
    semantic_expand = config.get("semantic_expand", True)

    score = 1.0
    details = []

    for word in must_contain:
        found = False
        # 先尝试精确匹配
        if word in answer:
            found = True
            match_type = "精确"
        else:
            # 尝试正则匹配（如果 word 包含正则元字符，用户可能有意使用正则）
            try:
                if re.search(word, answer):
                    found = True
                    match_type = "正则"
            except re.error:
                pass

        # 语义扩展：如果精确和正则都失败，尝试同义词/近义词
        if not found and semantic_expand:
            expanded = _semantic_expand(word)
            for syn in expanded:
                if syn in answer:
                    found = True
                    match_type = f"语义扩展({syn})"
                    break

        if found:
            details.append(f"✓ 包含必要关键词: '{word}' ({match_type}匹配)")
        else:
            details.append(f"✗ 缺少必要关键词: '{word}'")
            score -= 1.0 / max(len(must_contain), 1)

    for word in forbidden:
        if word in answer:
            details.append(f"✗ 包含禁用词: '{word}'")
            score = 0.0
        else:
            details.append(f"✓ 未包含禁用词: '{word}'")

    return max(0.0, score), "\n".join(details)


def _semantic_expand(word: str) -> List[str]:
    """对中文关键词进行简单的语义扩展"""
    expansions = {
        '量词': ['量词', '∀', '∃', '全称', '存在', 'quantifier'],
        '存在': ['存在', '∃', '有', '至少有一个'],
        '省略': ['省略', '隐含', '缺省', '省略了', '未明说'],
        '31%': ['31%', '31', '0.31', '三成', '约30%'],
        '循环': ['循环', 'cycle', '环', 'cycle length'],
        '信息': ['信息', 'information', 'bit', '比特', '信息论'],
        '快照': ['快照', 'snapshot', '一致性读', '快照读'],
        'MVCC': ['MVCC', '多版本', '版本链'],
        '斯坦纳': ['斯坦纳', 'Steiner', 'steiner'],
        '依存': ['依存', 'dependency', '句法树', '依存句法'],
        '使役': ['使役', 'causative', 'させる', '让'],
        '林远': ['林远', '少爷', '林远少爷'],
        '循环依赖': ['循环依赖', '循环', 'circular', '双向依赖'],
        'RESERVE_TIMEOUT': ['RESERVE_TIMEOUT', 'Reserve timeout', 'timeout=5', 'timeout 5'],
        'consul': ['consul', 'Consul', '服务发现'],
        '蕴含': ['蕴含', 'implication', '蕴涵', '推出'],
        '预设': ['预设', 'presupposition', '前提'],
        '杜甫': ['杜甫', '杜诗', '子美'],
        '王维': ['王维', '王诗', '摩诘'],
        '李商隐': ['李商隐', '李诗', '义山'],
        '安眠药': ['安眠药', '白色粉末', '粉末'],
        '纸条': ['纸条', '字条'],
        '密室': ['密室', '反锁'],
    }
    return expansions.get(word, [word])


def validate_python_exec(answer: str, code_to_run: str) -> Tuple[float, str]:
    """提取代码块与测试代码合并执行"""
    code_blocks = extract_code_blocks(answer)
    if not code_blocks:
        return 0.0, "未找到代码块"

    best_score = 0.0
    best_detail = "所有代码块均失败"

    for idx, user_code in enumerate(code_blocks):
        full_code = user_code + "\n\n" + code_to_run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        try:
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return 1.0, f"代码块 #{idx+1} 通过全部测试:\n{result.stdout.strip()[:500]}"
            else:
                # 分析错误类型
                stderr = result.stderr.strip()
                if 'AssertionError' in stderr:
                    detail = f"代码块 #{idx+1} 断言失败:\n{stderr[:600]}"
                elif 'NameError' in stderr:
                    detail = f"代码块 #{idx+1} 名称错误（变量/函数未定义）:\n{stderr[:600]}"
                elif 'SyntaxError' in stderr:
                    detail = f"代码块 #{idx+1} 语法错误:\n{stderr[:600]}"
                else:
                    detail = f"代码块 #{idx+1} 执行失败:\n{stderr[:600]}"
                if best_score < 0.5:
                    best_score = 0.0
                    best_detail = detail
        except subprocess.TimeoutExpired:
            best_detail = f"代码块 #{idx+1} 执行超时"
        except Exception as e:
            best_detail = f"代码块 #{idx+1} 执行异常: {str(e)}"
        finally:
            os.unlink(temp_path)

    return best_score, best_detail


def validate_code_fix(answer: str, config: Dict[str, Any]) -> Tuple[float, str]:
    """验证代码修复类答案（支持代码模式 + 文本语义检查）"""
    must_patterns = config.get("must_contain_patterns", [])
    forbidden_simple = config.get("forbidden_simple", [])
    must_contain = config.get("must_contain", [])
    forbidden_words = config.get("forbidden_words", [])

    code_score = 1.0
    code_details = []

    for pat in must_patterns:
        try:
            found = bool(re.search(pat, answer))
        except re.error:
            found = pat in answer
        if found:
            code_details.append(f"✓ 包含必要模式: '{pat}'")
        else:
            code_details.append(f"✗ 缺少必要模式: '{pat}'")
            code_score -= 0.3

    for word in forbidden_simple:
        if word in answer:
            code_details.append(f"✗ 包含禁用模式: '{word}'")
            code_score -= 0.5

    code_score = max(0.0, code_score)

    # 文本语义检查
    text_score = 1.0
    text_details = []
    for word in must_contain:
        found = False
        if word in answer:
            found = True
        else:
            try:
                if re.search(word, answer):
                    found = True
            except re.error:
                pass
        if not found:
            expanded = _semantic_expand(word)
            for syn in expanded:
                if syn in answer:
                    found = True
                    break
        if found:
            text_details.append(f"✓ 包含必要文本: '{word}'")
        else:
            text_details.append(f"✗ 缺少必要文本: '{word}'")
            text_score -= 1.0 / max(len(must_contain), 1)

    for word in forbidden_words:
        found = False
        try:
            if re.search(word, answer):
                found = True
        except re.error:
            pattern = r'(?<![A-Za-z0-9\u4e00-\u9fa5])' + re.escape(word) + r'(?![A-Za-z0-9\u4e00-\u9fa5])'
            if re.search(pattern, answer):
                found = True
        if found:
            text_details.append(f"✗ 包含禁用文本: '{word}'")
            text_score = 0.0
        else:
            text_details.append(f"✓ 未包含禁用文本: '{word}'")

    text_score = max(0.0, text_score)

    # 融合分数：代码占 70%，文本占 30%
    if must_contain or forbidden_words:
        final_score = 0.7 * code_score + 0.3 * text_score
        details = code_details + text_details
    else:
        final_score = code_score
        details = code_details

    return final_score, "\n".join(details)


VALIDATOR_MAP = {
    "math_range": validate_math_range,
    "exact_text": validate_exact_text,
    "python_exec": validate_python_exec,
    "code_fix": validate_code_fix,
    "none": lambda a, c: (None, "主观题，无客观验证"),
}


def run_objective_validation(test_case, answer: str) -> Dict[str, Any]:
    """根据 test_case.objective_type 调用对应验证器"""
    validator = VALIDATOR_MAP.get(test_case.objective_type)
    if validator is None:
        return {
            "type": test_case.objective_type,
            "score": None,
            "detail": f"未知验证器类型: {test_case.objective_type}",
        }

    if test_case.objective_type == "python_exec":
        score, detail = validator(answer, test_case.code_to_run or "")
    else:
        score, detail = validator(answer, test_case.objective_config)

    return {"type": test_case.objective_type, "score": score, "detail": detail}


# ==================== 代码块提取与运行验证 ====================

def extract_code_blocks(text: str):
    """从文本中提取代码块"""
    matches = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        return matches
    # 也尝试没有语言标记的代码块
    matches = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        return matches
    # 如果连 ``` 都没有，尝试提取缩进代码或 def/class 开头的代码段
    code_segments = re.findall(r'((?:^def\s+\w+|^class\s+\w+|^import\s+\w+).*?)(?=\n\n|\Z)', text, re.MULTILINE | re.DOTALL)
    if code_segments:
        return code_segments
    return []


def run_code_verification(code: str, test_code: str) -> dict:
    """合并代码与测试代码后执行"""
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name
    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Execution timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
    finally:
        os.unlink(temp_path)
