#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把院方健康教育资料（.docx/.doc）转换成 RAG 可索引的 Markdown。

用法：
  # 只提取到临时目录检查
  python3 tools/rag/import_sctcm_2025.py \
      --src "/home/artrc/文档/2025省中医院健康宣教" \
      --out /tmp/rag2025/extracted

  # 直接生成到 docs（会按 --prefix 命名，默认 2025_）
  python3 tools/rag/import_sctcm_2025.py --src ... --out tools/rag/docs

依赖：
  - .docx：Python 标准库（zipfile + ElementTree）
  - .doc ：antiword（Ubuntu: apt-get download antiword && dpkg-deb -x ...）
           也可用 --antiword 指定二进制路径，脚本自动设置 ANTIWORDHOME。

输出格式（与 tools/rag/docs 现有约定一致）：
  > 资料来源：2025省中医院健康宣教/<相对路径>
  # 标题
  ## 小节
  正文……
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 中文序号标题：一、 二、……；要求短句或是问句/冒号结尾，避免把长段落误判成标题
_CN_H2 = re.compile(r"^[一二三四五六七八九十]+、\s*(.+)$")
_CN_H3 = re.compile(r"^[（(][一二三四五六七八九十]+[）)]\s*(.+)$")
_NUM_H3 = re.compile(r"^\d+[.、．]\s*(.{0,28})$")


def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _is_heading(text: str, bold: bool = False, size: float = 0.0) -> int:
    """返回 0=正文 2=## 3=###。保守判定，宁少勿多。"""
    t = text.strip()
    if not t or len(t) > 60:
        return 0
    if _CN_H2.match(t) and (len(t) <= 45 or t.endswith(("：", "？", "?"))):
        return 2
    if _CN_H3.match(t) and len(t) <= 40:
        return 3
    if t.endswith(("：", ":")) and len(t) <= 24:
        # 无编号的“xxx：”通常是文档的一级小节（如“透析中要注意什么：”），
        # 带编号/括号的才是子标题
        if re.match(r"^[（(]?[0-9一二三四五六七八九十]+[）)、.．]", t):
            return 3
        return 2
    m = _NUM_H3.match(t)
    if m and t.rstrip().endswith(("：", ":", "？", "?")) and len(t) <= 30:
        return 3
    if bold and len(t) <= 40:
        return 3
    if size >= 14 and len(t) <= 40:
        return 3
    return 0


def _para_props(p):
    """返回 (bold, font_size_pt, alignment)。"""
    bold = False
    size = 0.0
    align = ""
    ppr = p.find(W + "pPr")
    if ppr is not None:
        jc = ppr.find(W + "jc")
        if jc is not None:
            align = jc.get(W + "val") or ""
    runs = p.findall(W + "r")
    for r in runs:
        rpr = r.find(W + "rPr")
        if rpr is None:
            continue
        b = rpr.find(W + "b")
        if b is not None and (b.get(W + "val") not in ("0", "false")):
            bold = True
        sz = rpr.find(W + "sz")
        if sz is not None:
            try:
                size = max(size, int(sz.get(W + "val")) / 2.0)
            except Exception:
                pass
    return bold, size, align


def _para_text(p) -> str:
    parts = []
    for node in p.iter():
        tag = node.tag
        if tag == W + "t":
            parts.append(node.text or "")
        elif tag == W + "tab":
            parts.append("\t")
        elif tag in (W + "br", W + "cr"):
            parts.append("\n")
    return "".join(parts)


def _docx_paragraphs(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    out = []
    if body is None:
        return out
    for p in body.findall(W + "p"):
        txt = _para_text(p)
        if not txt.strip():
            continue
        bold, size, align = _para_props(p)
        out.append((txt, bold, size, align))
    return out


def _run_antiword(path: str, antiword: str) -> str:
    home = os.environ.get("ANTIWORDHOME")
    if not home:
        cand = os.path.join(os.path.dirname(os.path.dirname(antiword)), "share", "antiword")
        if os.path.isdir(cand):
            home = cand
    env = dict(os.environ)
    if home:
        env["ANTIWORDHOME"] = home
    cmd = [antiword, "-w", "0", "-m", "UTF-8.txt", path]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"antiword 失败: {r.stderr.strip()[:200]}")
    return r.stdout


def _doc_paragraphs(path: str, antiword: str):
    raw = _run_antiword(path, antiword).replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    # 去页码/页眉：短行且重复出现两次以上
    from collections import Counter
    cnt = Counter(_clean_line(x) for x in lines if _clean_line(x))
    out = []
    for x in lines:
        t = _clean_line(x)
        if not t:
            out.append(("", False, 0.0, ""))
            continue
        if re.fullmatch(r"[-—–]?\s*\d+\s*[-—–]?", t):
            continue
        if cnt[t] >= 3 and len(t) <= 30:
            continue
        out.append((t, False, 0.0, ""))
    return out


def _to_markdown(paras, title_fallback: str, source_rel: str):
    """把段落列表转成 markdown。"""
    paras = [(t.strip(), b, s, a) for t, b, s, a in paras if t.strip()]
    if not paras:
        return f"> 资料来源：2025省中医院健康宣教/{source_rel}\n\n# {title_fallback}\n"
    # 标题取第一段；若第一段过长则用文件名
    title = paras[0][0]
    if len(title) > 40:
        title = title_fallback
    else:
        paras = paras[1:]
    lines = [f"> 资料来源：2025省中医院健康宣教/{source_rel}", "", f"# {title}", ""]
    h2_seen = False
    for text, bold, size, align in paras:
        t = _clean_line(text)
        if not t:
            lines.append("")
            continue
        level = _is_heading(t, bold, size)
        # 文档首个小节前若没有二级标题，保持正文
        if level == 3 and not h2_seen:
            level = 2
            h2_seen = True
        if level == 2:
            h2_seen = True
            lines += ["", f"## {t}", ""]
        elif level == 3:
            lines += ["", f"### {t}", ""]
        else:
            # 多行段落（w:br）按换行保留
            for seg in t.split("\n"):
                seg = _clean_line(seg)
                if seg:
                    lines.append(seg)
    # 压缩多余空行
    out = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[\\/:*?\"<>|,，。？?！!；;（）()【】\[\]\s]+", "_", stem).strip("_")
    return stem or "doc"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始资料目录（含子目录）")
    ap.add_argument("--out", required=True, help="输出 markdown 目录")
    ap.add_argument("--prefix", default="2025_", help="输出文件名前缀")
    ap.add_argument("--antiword", default=os.environ.get("ANTIWORD_BIN", "antiword"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not shutil.which(args.antiword) and not os.path.exists(args.antiword):
        print("[ERROR] 找不到 antiword，可用 --antiword /path/to/antiword 指定", file=sys.stderr)
        return 1

    files = []
    for root, _, names in os.walk(args.src):
        for n in sorted(names):
            if n.lower().endswith((".docx", ".doc")):
                files.append(os.path.join(root, n))
    files.sort()
    if not files:
        print("[ERROR] 源目录没有 .doc/.docx", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    ok = 0
    for path in files:
        rel = os.path.relpath(path, args.src)
        # 分类目录 + 文件名，如 一、首次透析_1.血液透析那些事
        cat = os.path.basename(os.path.dirname(path))
        stem = f"{cat}_{_safe_stem(path)}" if cat and cat != os.path.basename(args.src) else _safe_stem(path)
        out_name = f"{args.prefix}{stem}.md"
        out_path = os.path.join(args.out, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_name}")
            continue
        try:
            if path.lower().endswith(".docx"):
                paras = _docx_paragraphs(path)
            else:
                paras = _doc_paragraphs(path, args.antiword)
            md = _to_markdown(paras, _safe_stem(path), rel.replace(os.sep, "/"))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
            ok += 1
            print(f"[ok] {out_name} ({len(md)}字)")
        except Exception as exc:
            print(f"[ERR] {rel}: {exc}", file=sys.stderr)
    print(f"完成 {ok}/{len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
