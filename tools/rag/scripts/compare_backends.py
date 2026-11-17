#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 A/B：同一批问题对比「旧百炼应用」与「新 RAG+直连 qwen-plus」。

输出 records: tools/rag/scripts/out/compare_backends.{json,md}
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "llm_node_py", "llm_node_py"))

from get_dashscope_key import get_dashscope_key  # noqa: E402
from dashscope import Application  # noqa: E402
from openai import OpenAI  # noqa: E402

APP_ID = "b7ed8dad5c0c483d8e865a3be0b4e088"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus-latest"

MED_SYSTEM_PROMPT = """你是血透中心巡诊机器人「小医」，用轻柔亲切的语气和患者对话。
必须遵守：
1. 只能依据【参考资料】回答医学与护理问题。资料里没有的内容，直接说“这个问题我资料里没有，建议您咨询护士或医生”，不要编造。
2. 口语化短句，一次说 1-3 句，总长不超过 80 字，适合语音播报；不要列出资料编号。
3. 不给具体用药剂量、不诊断；涉及具体数值时只做一般性提醒，并建议咨询医生。
4. 患者出现胸痛、呼吸困难、意识改变、内瘘出血不止、明显心慌乏力等紧急情况时，立即说“我马上帮您呼叫护士”，不要展开长篇内容。
5. 禁止输出表情符号、动作描写；回答中不要出现“着急”两个字。"""

OLD_PERSONA_USER = (
    "有几个血透中心的病人，现在在血透中心做透析，你是巡诊的机器人，名字叫小医。"
    "你的大脑跟病人的测量数据是实时联系在一起的，下面几个病人开始对话。"
    "对话的时候不要长篇大论，像真人一样，使用对话句子，也就是我们一句来一句去这样。"
    "语气需要轻柔舒缓。所有使用单位名称转化为中文。尽可能不出现英文。"
    "你只需要监测，禁止输出执行实际动作,例如倒水。不要增加小动作。不要输出表情符号。"
    "在需要时，给出数据偏高或偏低，不需要具体数据。不要给出实际的执行动作，"
    "而是不用通过类似读取的方式。讲话必须亲切，温柔。科普类的内容可以丰富一些。"
    "叫护士或医生时回答好的，我马上帮您联系护士或医生。您先安心休息一会儿。"
    "禁止在回答中出现着急两个字。问题不严重时，你来提出解决办法，不要经常输出寻找医生。"
    "询问天气/温度等关键词，仅输出杭州本地核心信息."
)
OLD_PERSONA_BOT = (
    "好的，我记住了,我的名字是小医，我会采用真人对话风格那样进行对话，"
    "并使用轻柔温和的语气，我不会输出执行动作，不会输出表情符合，"
    "不会给出具体的数值，问题不严重时我会自己提出解决办法不会寻求医生的帮助，"
    "不会使用'着急'等急迫性质的词语。"
)
REFUSAL_WORDS = ["没有", "资料", "建议", "咨询", "护士", "医生", "不知道", "不清楚"]


def _clean_proxies():
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)


def call_old_app(question):
    t0 = time.perf_counter()
    first = None
    text = ""
    for resp in Application.call(
        api_key=get_dashscope_key(), app_id=APP_ID, prompt=question,
        history=[{"user": OLD_PERSONA_USER, "bot": OLD_PERSONA_BOT}],
        stream=True, incremental_output=True,
    ):
        if resp.status_code == 200:
            chunk = resp.output.text
            if chunk and first is None:
                first = time.perf_counter() - t0
            text += chunk
        else:
            raise RuntimeError(f"app error: {resp}")
    return {"answer": text.strip(), "ttft_ms": (first or 0) * 1000,
            "total_ms": (time.perf_counter() - t0) * 1000}


def rag_retrieve(rag_url, query, top_k=4, timeout=5.0):
    url = rag_url.rstrip("/")
    if not url.endswith("/retrieve"):
        url += "/retrieve"
    payload = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_user_prompt(question, chunks):
    if not chunks:
        return (f"【参考资料】\n（本轮没有检索到相关资料）\n\n【患者问题】{question}\n\n"
                "注意：没有资料时不要编造，按系统规则回答（说明资料里没有、建议咨询护士或医生）。")
    lines = ["【参考资料】"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}]（来源：{c.get('title') or c.get('doc_id')}）"
                     f"{(c.get('text') or '').strip().replace(chr(10), ' ')}")
    lines += ["", f"【患者问题】{question}", "",
              "注意：只依据上述资料回答；资料和问题无关时按系统规则说明资料里没有。"]
    return "\n".join(lines)


def call_new_rag(question, rag_url, client, top_k=4):
    t0 = time.perf_counter()
    rag = rag_retrieve(rag_url, question, top_k=top_k)
    rag_ms = (time.perf_counter() - t0) * 1000
    chunks = rag.get("chunks") or []
    messages = [
        {"role": "system", "content": MED_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True,
        temperature=0.6, top_p=0.8, max_tokens=400)
    text, first = "", None
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if not content:
            continue
        if first is None:
            first = time.perf_counter() - t0
        text += content
    return {"answer": text.strip(), "ttft_ms": (first or 0) * 1000,
            "total_ms": (time.perf_counter() - t0) * 1000, "rag_ms": rag_ms,
            "rag_titles": [c.get("title") or c.get("doc_id") for c in chunks],
            "rag_ids": [c.get("id") for c in chunks]}


def bigrams(text):
    c = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")
    return {c[i:i + 2] for i in range(len(c) - 1)}


def point_coverage(answer, points):
    ans = bigrams(answer)
    if not ans or not points:
        return 0.0, []
    covered = []
    for p in points:
        pb = bigrams(p)
        if pb and len(pb & ans) / len(pb) >= 0.6:
            covered.append(p)
    return len(covered) / len(points), covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-url", default="http://127.0.0.1:8765")
    ap.add_argument("--eval", default=os.path.join(HERE, "..", "eval", "eval_set.json"))
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    _clean_proxies()
    client = OpenAI(api_key=get_dashscope_key(), base_url=BASE_URL)
    items = json.load(open(args.eval, encoding="utf-8"))[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for idx, item in enumerate(items, 1):
        q = item["question"]
        print(f"[{idx}/{len(items)}] {q}", flush=True)
        row = {"id": item.get("id"), "type": item.get("type"), "question": q,
               "expected_points": item.get("expected_points") or []}
        try:
            row["old_app"] = call_old_app(q)
        except Exception as exc:  # noqa: BLE001
            row["old_app"] = {"answer": f"[ERROR] {exc}", "ttft_ms": None, "total_ms": None}
        try:
            row["new_rag"] = call_new_rag(q, args.rag_url, client, args.top_k)
        except Exception as exc:  # noqa: BLE001
            row["new_rag"] = {"answer": f"[ERROR] {exc}", "ttft_ms": None,
                              "total_ms": None, "rag_titles": []}
        for key in ("old_app", "new_rag"):
            ans = row[key].get("answer") or ""
            cov, covered = point_coverage(ans, row["expected_points"])
            row[key]["point_coverage"] = round(cov, 3)
            row[key]["covered_points"] = covered
            row[key]["refusal_like"] = any(w in ans for w in REFUSAL_WORDS)
            row[key]["len"] = len(ans)
        results.append(row)

    with open(os.path.join(args.out_dir, "compare_backends.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    def fmt(x):
        return f"{x:.0f}" if isinstance(x, (int, float)) else "—"

    lines = ["# Phase 0 答案对比：旧百炼应用 vs 新 RAG + qwen-plus 直连\n",
             "| # | 问题 | 类型 | 旧 首块/总 (ms) | 新 检索/首块/总 (ms) | 旧要点覆盖 | 新要点覆盖 |",
             "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for i, r in enumerate(results, 1):
        o, n = r["old_app"], r["new_rag"]
        lines.append(
            f"| {i} | {r['question'][:26]} | {r['type']} | "
            f"{fmt(o.get('ttft_ms'))}/{fmt(o.get('total_ms'))} | "
            f"{fmt(n.get('rag_ms'))}/{fmt(n.get('ttft_ms'))}/{fmt(n.get('total_ms'))} | "
            f"{o.get('point_coverage')} | {n.get('point_coverage')} |")
    lines += ["", "## 逐题答案\n"]
    for i, r in enumerate(results, 1):
        o, n = r["old_app"], r["new_rag"]
        lines += [
            f"### {i}. {r['question']}（{r['type']}）", "",
            f"- 期望要点：{'；'.join(r['expected_points'])}",
            f"- 检索命中：{n.get('rag_titles')}", "",
            f"**旧（百炼应用）**[{fmt(o.get('ttft_ms'))}ms / {fmt(o.get('total_ms'))}ms]", "",
            f"> {o.get('answer', '')}", "",
            f"**新（RAG+直连）**[检索 {fmt(n.get('rag_ms'))}ms / 首块 {fmt(n.get('ttft_ms'))}ms / "
            f"总 {fmt(n.get('total_ms'))}ms]", "",
            f"> {n.get('answer', '')}", ""]
    md_path = os.path.join(args.out_dir, "compare_backends.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    def avg(key, field):
        vals = [r[key][field] for r in results if isinstance(r[key].get(field), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== 汇总（均值） ===")
    print(f"旧 app     : TTFT {avg('old_app','ttft_ms'):.0f}ms, 总 {avg('old_app','total_ms'):.0f}ms, "
          f"要点覆盖 {avg('old_app','point_coverage'):.2f}")
    print(f"新 RAG+直连: 检索 {avg('new_rag','rag_ms'):.0f}ms, TTFT {avg('new_rag','ttft_ms'):.0f}ms, "
          f"总 {avg('new_rag','total_ms'):.0f}ms, 要点覆盖 {avg('new_rag','point_coverage'):.2f}")
    print(f"\n已保存: {md_path}")


if __name__ == "__main__":
    main()
