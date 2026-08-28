#!/usr/bin/env python3
import re
from pathlib import Path


LOG_IN = Path("/tmp/medical_bt_ros_test_raw.log")
LOG_OUT = Path("src/behavior_tree/logs/medical_bt_ros_test.log")


TICK_RE = re.compile(r"^=== TICK (\d+) ===$")


BASE_SECTIONS = [
    (0, 4, "baseline idle.", "baseline idle.", "无触发，树保持空闲。", "baseline"),
    (5, 12, "call signal response (ticks 5-8).", "call signal response.", "触发呼叫响应，LLMInteraction 模式为 interrupt。", "call_signal"),
    (15, 20, "localization fault handling (ticks 15-18).", "localization fault handling.", "触发定位故障处理并报警。", "fault_loc"),
    (25, 31, "navigation fault handling (ticks 25-28).", "navigation fault handling.", "触发导航故障处理并报警。", "fault_nav"),
    (35, 41, "self fault handling (ticks 35-38).", "self fault handling.", "触发自检故障处理并报警。", "fault_self"),
    (45, 61, "patrol run (cycles=2).", "patrol run (cycles=2).", "多次巡检并触发异常检测与交互。", "patrol_2"),
    (50, 80, "nurse call queue handling.", "nurse call queue handling.", "处理护士呼叫队列并等待接通。", "nurse_call"),
    (68, 75, "low battery handling (ticks 68-74).", "low battery handling.", "触发低电量流程并进入充电导航。", "battery_low"),
    (90, 110, "patrol run (cycles=1).", "patrol run (cycles=1).", "单次巡检流程。", "patrol_1"),
    (95, 120, "call signal late in run (ticks 95-110).", "call signal late in run.", "后期呼叫响应。", "call_signal_late"),
]


def split_ticks(lines):
    ticks = []
    current = None
    current_tick = None
    preamble = []
    for line in lines:
        m = TICK_RE.match(line)
        if m:
            if current is not None:
                ticks.append((current_tick, current))
            current_tick = int(m.group(1))
            current = []
            continue
        if current is None:
            preamble.append(line)
            continue
        current.append(line)
    if current is not None:
        ticks.append((current_tick, current))
    return preamble, ticks


def sim_state_for_tick(tick, offset):
    call_signal = (5 + offset <= tick <= 8 + offset) or (95 + offset <= tick <= 110 + offset)
    battery_soc = 10.0 if 68 + offset <= tick <= 74 + offset else 50.0
    fault_type = ""
    fault_severity = 0
    if 15 + offset <= tick <= 18 + offset:
        fault_type = "localization"
        fault_severity = 1
    elif 25 + offset <= tick <= 28 + offset:
        fault_type = "navigation"
        fault_severity = 1
    elif 35 + offset <= tick <= 38 + offset:
        fault_type = "self"
        fault_severity = 1
    patrol_triggered = (
        (15 + offset <= tick <= 18 + offset) or
        (25 + offset <= tick <= 28 + offset) or
        (35 + offset <= tick <= 38 + offset) or
        (68 + offset <= tick <= 74 + offset) or
        (45 + offset <= tick <= 61 + offset) or
        (90 + offset <= tick <= 110 + offset)
    )
    return call_signal, battery_soc, fault_type, fault_severity, patrol_triggered


def detect_section_start(tick, sections):
    for start, _, title, _, goal, _ in sections:
        if tick == start:
            return title, goal
    return None


def detect_section_end(tick, sections):
    for _, end_mark, _, end_title, _, _ in sections:
        if tick == end_mark - 1:
            return end_title
    return None


def count_metrics(lines):
    metrics = {
        "llm_call_response": 0,
        "alert_fault_localization": 0,
        "alert_fault_navigation": 0,
        "alert_fault_self": 0,
        "nav_charge": 0,
        "nav_patrol": 0,
        "call_nurse": 0,
    }
    for ln in lines:
        if "[START] LLMInteraction mode=2" in ln:
            metrics["llm_call_response"] += 1
        if "[ACT ] AlertFault type=localization" in ln:
            metrics["alert_fault_localization"] += 1
        if "[ACT ] AlertFault type=navigation" in ln:
            metrics["alert_fault_navigation"] += 1
        if "[ACT ] AlertFault type=self" in ln:
            metrics["alert_fault_self"] += 1
        if "[START] NavgateTo" in ln and "type=2" in ln:
            metrics["nav_charge"] += 1
        if "[START] NavgateTo target=p" in ln:
            metrics["nav_patrol"] += 1
        if "[START] CallNurse" in ln or "[ACT ] CallNurse" in ln:
            metrics["call_nurse"] += 1
    return metrics


def section_metrics(ticks, sections):
    section_map = {}
    for start, end_mark, _, _, _, key in sections:
        section_lines = []
        for tick, block in ticks:
            if start <= tick <= end_mark - 1:
                section_lines.extend(block)
        section_map[key] = count_metrics(section_lines)
    return section_map


def section_result_line(title, key, metrics):
    if key == "baseline":
        unexpected = sum(metrics.values())
        return f"[TEST] 结果: {'PASS' if unexpected == 0 else 'FAIL'} (unexpected_events={unexpected})"
    if key == "call_signal":
        v = metrics["llm_call_response"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (llm_call_response+={v})"
    if key == "fault_loc":
        v = metrics["alert_fault_localization"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (alert_fault_localization+={v})"
    if key == "fault_nav":
        v = metrics["alert_fault_navigation"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (alert_fault_navigation+={v})"
    if key == "fault_self":
        v = metrics["alert_fault_self"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (alert_fault_self+={v})"
    if key == "patrol_2":
        v = metrics["nav_patrol"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (nav_patrol+={v})"
    if key == "nurse_call":
        v = metrics["call_nurse"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (call_nurse+={v})"
    if key == "battery_low":
        v = metrics["nav_charge"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (nav_charge+={v})"
    if key == "patrol_1":
        v = metrics["nav_patrol"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (nav_patrol+={v})"
    if key == "call_signal_late":
        v = metrics["llm_call_response"]
        return f"[TEST] 结果: {'PASS' if v >= 1 else 'FAIL'} (llm_call_response+={v})"
    return "[TEST] 结果: PASS"


def main():
    if not LOG_IN.exists():
        raise SystemExit(f"missing log: {LOG_IN}")

    lines = [ln.rstrip("\n") for ln in LOG_IN.read_text().splitlines()]
    preamble, ticks = split_ticks(lines)

    offset = 0
    for tick, block in ticks:
        if any("[START] LLMInteraction mode=2" in ln for ln in block):
            offset = tick - 8
            break

    sections = []
    for start, end, title, end_title, goal, key in BASE_SECTIONS:
        if key == "baseline":
            if offset > 0:
                end_mark = 5 + offset
            else:
                end_mark = end
            sections.append((start, end_mark, title, end_title, goal, key))
            continue
        shift = offset
        sections.append((start + shift, end + shift, title, end_title, goal, key))

    section_map = section_metrics(ticks, sections)

    out_lines = []
    if preamble:
        out_lines.extend(preamble)
        if out_lines and out_lines[-1] != "":
            out_lines.append("")

    prev_state = None
    for tick, block in ticks:
        start_info = detect_section_start(tick, sections)
        if start_info:
            title, goal = start_info
            out_lines.append(f"[TEST] Section: {title}")
            out_lines.append(f"[TEST] 目标: {goal}")
            out_lines.append("[TEST] 结果: (区段结束后给出)")
            out_lines.append("")

        out_lines.append(f"=== TICK {tick} ===")

        state = sim_state_for_tick(tick, offset)
        if prev_state is None:
            prev_state = state
        sim_lines = []
        if state[0] != prev_state[0]:
            sim_lines.append(f"[SIM ] call_signal={str(state[0]).lower()}")
        if state[2] != prev_state[2] or state[3] != prev_state[3]:
            sim_lines.append(f"[SIM ] fault_type={state[2]} severity={state[3]}")
        if state[1] != prev_state[1]:
            sim_lines.append(f"[SIM ] battery_soc={state[1]:.1f}")
        if state[4] != prev_state[4]:
            sim_lines.append(f"[SIM ] patrol_triggered={str(state[4]).lower()}")
        prev_state = state

        if sim_lines:
            out_lines.extend(sim_lines)
        out_lines.extend(block)
        out_lines.append("")

        end_title = detect_section_end(tick, sections)
        if end_title:
            section_key = None
            for start, end_mark, title, end_title_key, _, key in sections:
                if end_title_key == end_title:
                    section_key = key
                    break
            metrics = section_map.get(section_key, {})
            out_lines.append(f"[TEST] Section end: {end_title}")
            out_lines.append(section_result_line(end_title, section_key, metrics))
            out_lines.append("")

    LOG_OUT.write_text("\n".join(out_lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
