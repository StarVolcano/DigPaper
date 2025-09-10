import asyncio
import cgi
import os
import time
import shutil
import uuid
import json
import html
from asyncio import CancelledError
from pathlib import Path
import typing as T
import requests

from datetime import datetime
import math
import fitz  # PyMuPDF for PDF annotation
import re

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from pypdf import PdfMerger

import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from gradio import Progress

logger = logging.getLogger(__name__)

BABELDOC_MODEL = OnnxModel.load_available()
UPLOAD_DIR = Path("uploadfile")
UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATION_MAP = {}  # 全局存储标注映射

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))

##################################################################
# TODO: 全文分析后端API调用占位符 - 实际实现时需要替换为真实的后端调用
def call_backend_analysis(pdf_file_path: str) -> Dict[str, str]:
    """
    调用后端分析服务，返回两个JSON文件的路径
    """

    url = "http://localhost:8004/process"
    files = {
        "file": open(pdf_file_path, "rb"),  # 替换为你的 PDF 文件路径
    }
    response = requests.post(url, files=files)
    print(response)
    print(response.json())
    
    return response.json()

def save_uploaded_file(file_path: str) -> str:
    """
    保存上传的文件到uploadfile目录
    """
    try:
        file_name = os.path.basename(file_path)
        saved_path = UPLOAD_DIR / file_name

        # 如果文件已存在，添加后缀
        counter = 1
        while saved_path.exists():
            name, ext = os.path.splitext(file_name)
            saved_path = UPLOAD_DIR / f"{name}_{counter}{ext}"
            counter += 1

        shutil.copy2(file_path, saved_path)
        return str(Path(saved_path).resolve())  # 绝对路径
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        return file_path

def F_analyze_paper(pdf_file_path: str) -> str:
    """处理PDF论文，生成包含分析结果的新PDF"""
    try:
        if not pdf_file_path or not os.path.exists(pdf_file_path):
            raise FileNotFoundError(f"No valid PDF selected: {pdf_file_path}")
        # 保存上传的文件
        saved_pdf_path = save_uploaded_file(pdf_file_path)

        # 调用后端API处理PDF文件
        analysis_results = call_backend_analysis(saved_pdf_path)

        # 加载JSON文件内容
        def _safe_read_json(path_like: str):
            # 绝对路径直接用；相对路径就按工作目录和脚本目录各尝试一次
            here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
            cands = [path_like, os.path.join(os.getcwd(), path_like), os.path.join(here, path_like)]
            for c in cands:
                if os.path.exists(c):
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
            raise FileNotFoundError(f"JSON not found: {path_like}")

        full_analysis = _safe_read_json(analysis_results['full_analysis'])
        figure_analysis = _safe_read_json(analysis_results['figure_analysis'])

        # ====== 第一部分（全文分析） ======
        # ====== 从 JSON 取数据（保留变量名，不改 JSON 结构） ======
        title_text = full_analysis.get("title", "").strip()
        university_text = full_analysis.get("university", "").strip()
        abstract_text = full_analysis.get("abstruct", "").strip()  # 注意：键名为 abstruct
        key_problem_md = full_analysis.get("关键问题", "").strip()
        key_methods = full_analysis.get("关键做法", [])  # list[{"步骤":..., "详细描述":...}, ...]

        # ====== 准备输出路径 ======
        base_name = os.path.splitext(os.path.basename(saved_pdf_path))[0]
        out_pdf_path = str((UPLOAD_DIR / f"{base_name}_analysis.pdf").resolve())

        def try_register_ttf(name: str, paths: list) -> bool:
            for p in paths:
                if os.path.exists(p):
                    try:
                        pdfmetrics.registerFont(TTFont(name, p))
                        return True
                    except Exception:
                        pass
            return False

        # 尝试注册 SimSun / SimHei，如果失败则回退到 CID 字体
        song_ok = try_register_ttf("SimSun", [
            "/home/huxc/paper_agent/front/simsun.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ])
        hei_ok = try_register_ttf("SimHei", [
            "/home/huxc/paper_agent/front/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ])

        if not song_ok:
            # ReportLab 自带的中文 CID 字体（无需本地 TTF）
            try:
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                song_font_name = 'STSong-Light'
            except Exception:
                song_font_name = 'Helvetica'  # 最后兜底
        else:
            song_font_name = 'SimSun'

        if not hei_ok:
            # 没有黑体就用宋体加粗来近似
            heiti_fallback = song_font_name
            heiti_is_real = False
        else:
            heiti_fallback = 'SimHei'
            heiti_is_real = True

        # ====== 样式（中文字号：三号=16pt；四号=14pt；小四=12pt；五号=10.5pt） ======
        def em_indent(font_size_pt: float, n_em: float = 2.0) -> float:
            return n_em * font_size_pt  # 近似 1em=字体大小

        styles = getSampleStyleSheet()
        style_title = ParagraphStyle(
            name="Title", parent=styles["Normal"], alignment=TA_CENTER,
            fontName=heiti_fallback, fontSize=16, leading=20, spaceAfter=10
        )
        style_univ = ParagraphStyle(
            name="University", parent=styles["Normal"], alignment=TA_CENTER,
            fontName=song_font_name, fontSize=10.5, leading=14, spaceAfter=10
        )
        style_section_h = ParagraphStyle(
            name="SectionH", parent=styles["Normal"], alignment=TA_CENTER,
            fontName=heiti_fallback, fontSize=14, leading=18, spaceBefore=8, spaceAfter=6
        )
        style_step_h = ParagraphStyle(
            name="StepH", parent=styles["Normal"], alignment=TA_LEFT,
            fontName=heiti_fallback, fontSize=12, leading=16, spaceBefore=6, spaceAfter=4
        )
        style_body = ParagraphStyle(
            name="Body", parent=styles["Normal"], alignment=TA_JUSTIFY,
            fontName=song_font_name, fontSize=10.5, leading=16,
            firstLineIndent=em_indent(10.5, 2.0), spaceAfter=4
        )
        style_abstract = ParagraphStyle(
            name="Abstract", parent=styles["Normal"], alignment=TA_JUSTIFY,
            fontName=song_font_name, fontSize=10.5, leading=16,
            spaceAfter=8
        )

        # ====== Markdown → ReportLab <para>（严格，含公式保护） ======
        M_INLINE = "§MATH_INLINE_%d§"
        M_BLOCK = "§MATH_BLOCK_%d§"

        def _extract_math(text: str):
            """抽出 $$...$$ 与 $...$，返回 (占位文本, 恢复列表)"""
            blocks = []
            inlines = []

            def repl_block(m):
                idx = len(blocks)
                blocks.append(m.group(0))  # 含 $$
                return M_BLOCK % idx

            def repl_inline(m):
                idx = len(inlines)
                inlines.append(m.group(0))  # 含 单个 $
                return M_INLINE % idx

            # 先处理 $$...$$（跨行）
            t = re.sub(r"\$\$(.+?)\$\$", repl_block, text, flags=re.S)
            # 再处理 $...$（行内），避免 \$ 误匹配
            t = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", repl_inline, t, flags=re.S)
            return t, blocks, inlines

        def _restore_math(t: str, blocks, inlines):
            # 用等宽字体包裹；保持原样字符（不渲染 LaTeX，仅保护与可读）
            for i, raw in enumerate(blocks):
                body = raw.strip("$")
                t = t.replace(M_BLOCK % i, f'<br/><font face="Courier">{body}</font><br/>')
            for i, raw in enumerate(inlines):
                body = raw.strip("$")
                t = t.replace(M_INLINE % i, f'<font face="Courier">{body}</font>')
            return t

        def md_to_rl_strict(text: str) -> str:
            """
            严格按 Markdown：单换行 -> <br/>；空行 -> 段落间距；支持 ** * ~~ ` 链接 列表 等；
            保护 $...$ / $$...$$，避免被其它规则破坏。
            """
            if not text:
                return ""
            t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

            # 0) 先保护公式
            t, mblocks, minlines = _extract_math(t)

            # 1) 链接 [text](url)  -> <link href="...">text</link>
            t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2">\1</link>', t)

            # 2) 列表行首："- " → 项目前缀符号（仅作用于行首）
            t = re.sub(r"(?m)^[ \t]*- +", "&#8226; ", t)

            # 3) 强/斜体/删除线/行内代码
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
            t = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", t)
            t = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', t)

            # 4) 特殊符号与脚注
            t = re.sub(r"\[\^(\d+)\^\]", r"[\1]", t)  # [^1^] -> [1]
            t = t.replace("^^", "^")
            t = t.replace("\\\\", "&#92;&#92;").replace("\\", "&#92;")

            # 5) 换行：单换行 -> <br/>；>=2 换行 -> <br/><br/>
            t = re.sub(r"\n{2,}", "<br/><br/>", t)
            t = t.replace("\n", "<br/>")

            # 6) 清理孤立星号
            t = t.replace("**", "")
            t = re.sub(r"(?<![a-zA-Z0-9<])\*(?![a-zA-Z0-9>])", "", t)

            # 7) 恢复公式
            t = _restore_math(t, mblocks, minlines)
            return t

        def _split_paragraphs(md_text: str):
            """按空行分段，段内保留单行换行（由 md_to_rl_strict 处理为 <br/>）。"""
            if not md_text:
                return []
            parts = re.split(r"\n\s*\n+", md_text.strip())
            return [p.strip() for p in parts if p.strip()]

        def md_to_plain(md: str) -> str:
            """
            将 markdown 简化为可搜索的纯文本：
            - 去 **粗体** / *斜体* / ~~删除线~~ / `code`
            - 链接 [text](url) -> text；图片 ![alt](url) -> alt
            - 公式去 $ 符号，仅保留内容
            - 去 'Source N:' 前缀，合并空白
            """
            if not md:
                return ""
            s = md.replace("\r\n", "\n").replace("\r", "\n")

            # 图片、链接
            s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
            s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

            # 代码
            s = re.sub(r"`{3}[\s\S]*?`{3}", " ", s)
            s = re.sub(r"`([^`]+)`", r"\1", s)

            # 粗斜体/删除线
            s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
            s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)
            s = re.sub(r"~~(.+?)~~", r"\1", s)

            # 公式内容保留，去定界符
            s = re.sub(r"\$\$(.+?)\$\$", r"\1", s, flags=re.S)
            s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", r"\1", s, flags=re.S)

            # 列表符号
            s = re.sub(r"(?m)^[ \t]*- +", "", s)

            # 去 Source 前缀
            s = re.sub(r"^Source\s*\d+\s*:\s*", "", s.strip(), flags=re.I)

            # 合并空白
            s = re.sub(r"[ \t]+", " ", s)
            s = re.sub(r"\n{2,}", "\n", s).strip()
            return s

        def norm_source_key(md: str) -> str:
            """用于去重/编号的 key：纯文本 + 压缩空白"""
            s = md_to_plain(md)
            return re.sub(r"\s+", " ", s).strip()

        # =========================
        # A) 统一参考段落编号（第一部分的再处理）
        # =========================
        def assign_gid_for_source(md_src: str, pool: dict, order_list: list) -> int:
            key = norm_source_key(md_src) or f"__EMPTY_{len(pool)+1}"
            if key not in pool:
                gid = len(pool) + 1
                pool[key] = gid
                order_list.append((gid, key, md_src, md_to_plain(md_src)))
            return pool[key]

        def relabel_citations_in_text(md_text: str, src_list_md: list, pool: dict, order_list: list) -> str:
            """把正文中的 [k]（局部）替换为 [gid]（全局）"""
            if not md_text:
                return md_text
            def _repl(m):
                try:
                    k = int(m.group(1))
                except:
                    return m.group(0)
                if 1 <= k <= len(src_list_md):
                    gid = assign_gid_for_source(src_list_md[k-1], pool, order_list)
                    return f"[{gid}]"
                return m.group(0)
            return re.sub(r"\[(\d+)\]", _repl, md_text)

        # 1) 先按出现顺序在正文中建立全局编号
        global_pool = {}            # key -> gid
        ordered_sources = []        # [(gid, key, raw_md, plain_txt)]

        key_methods_render = []
        if isinstance(key_methods, list) and key_methods:
            for item in key_methods:
                step_title_raw = str(item.get("步骤", "")).strip()
                detail_md_raw  = str(item.get("详细描述", "")).strip()
                src_list_md    = item.get("参考段落", []) or []
                detail_md_new  = relabel_citations_in_text(detail_md_raw, src_list_md, global_pool, ordered_sources)
                key_methods_render.append({
                    "步骤": step_title_raw,
                    "详细描述_统一编号": detail_md_new,
                    "参考段落": src_list_md
                })

        key_conclusions_src = full_analysis.get("关键结论", []) or []
        key_conclusions_render = []
        if isinstance(key_conclusions_src, list) and key_conclusions_src:
            for item in key_conclusions_src:
                concl_title_raw  = str(item.get("结论", "")).strip()
                concl_detail_raw = str(item.get("详细阐述", "")).strip()
                src_list_md      = item.get("参考段落", []) or []
                concl_detail_new = relabel_citations_in_text(concl_detail_raw, src_list_md, global_pool, ordered_sources)
                key_conclusions_render.append({
                    "结论": concl_title_raw,
                    "详细阐述_统一编号": concl_detail_new,
                    "参考段落": src_list_md
                })

        # 2) 补足未在正文中出现但存在于参考列表的段落，保证编号连续覆盖所有不同段落（例如 35 个）
        def add_missing_sources(groups):
            for g in groups:
                for md_src in g:
                    assign_gid_for_source(md_src, global_pool, ordered_sources)

        add_missing_sources([km["参考段落"] for km in key_methods_render])
        add_missing_sources([kc["参考段落"] for kc in key_conclusions_render])

        # 排序备用
        ordered_by_gid = sorted(ordered_sources, key=lambda x: x[0])  # (gid, key, raw_md, plain_txt)

        # ====== 组装“第一部分” ======
        story = []

        # 1) 标题（居中 黑体三号）
        if title_text:
            story.append(Paragraph("论文阅读报告", style_title))
            story.append(Paragraph(md_to_rl_strict(title_text), style_title))

        # 2) 学校（居中 宋体5号）
        if university_text:
            story.append(Paragraph(md_to_rl_strict(university_text), style_univ))

        # 3) 摘要（“摘 要：”黑体；内容宋体；均为5号；“摘”“要”之间全角空格）
        if abstract_text:
            abstract_label = f'<font name="{heiti_fallback}">摘\u3000要：</font>'
            abstract_body = f'<font name="{song_font_name}">{md_to_rl_strict(abstract_text)}</font>'
            story.append(Paragraph(abstract_label + abstract_body, style_abstract))

        # 4) 关键问题（居中 黑体四号；正文每段缩进2字宽）
        if key_problem_md:
            story.append(Paragraph("关键问题", style_section_h))
            for para in _split_paragraphs(key_problem_md):
                story.append(Paragraph(md_to_rl_strict(para), style_body))

        # 5) 关键做法：居中黑体四号；“步骤x 标题”黑体小四（去尾标点）；正文每段缩进2字宽
        if key_methods_render:
            story.append(Spacer(1, 6))
            story.append(Paragraph("关键做法", style_section_h))
            for idx, item in enumerate(key_methods_render, 1):
                step_title_raw = item["步骤"]
                short_title = step_title_raw.split("：", 1)[0].split(":", 1)[0]
                short_title = re.sub(r"^\*\*|\*\*$", "", short_title)
                short_title = re.sub(r"[：:。．．.\s]+$", "", short_title)
                step_h_line = f"步骤{idx} {short_title}"
                story.append(Paragraph(md_to_rl_strict(step_h_line), style_step_h))
                detail_md_new = item["详细描述_统一编号"]
                if detail_md_new:
                    for para in _split_paragraphs(detail_md_new):
                        story.append(Paragraph(md_to_rl_strict(para), style_body))

        # 6) 关键结论：居中黑体四号；“结论x 标题”黑体小四；正文每段缩进2字宽
        if key_conclusions_render:
            story.append(Spacer(1, 6))
            story.append(Paragraph("关键结论", style_section_h))
            for idx, item in enumerate(key_conclusions_render, 1):
                concl_title_raw = item["结论"]
                concl_short = re.sub(r"^\*\*|\*\*$", "", concl_title_raw)
                concl_short = re.sub(r"[：:。．．.\s]+$", "", concl_short)
                concl_h_line = f"结论{idx} {concl_short}"
                story.append(Paragraph(md_to_rl_strict(concl_h_line), style_step_h))
                concl_detail_new = item["详细阐述_统一编号"]
                if concl_detail_new:
                    for para in _split_paragraphs(concl_detail_new):
                        story.append(Paragraph(md_to_rl_strict(para), style_body))

        # ====== 第二部分（图表与表格综述等） ======
        # 正文样式沿用你已定义的宋体五号，不做首行缩进（与示例一致）
        style_md_body = ParagraphStyle(
            name="MdBody", parent=styles["Normal"], alignment=TA_JUSTIFY,
            fontName=song_font_name, fontSize=10.5, leading=16,
            spaceBefore=6, spaceAfter=12
        )

        # 页面宽度内等比缩放图片
        def make_scaled_image(img_path: str, max_w: float, max_h: float):
            try:
                ir = ImageReader(img_path)
                iw, ih = ir.getSize()
                if iw <= 0 or ih <= 0:
                    return None
                scale = min(max_w / float(iw), max_h / float(ih), 1.0)
                w, h = iw * scale, ih * scale
                if w <= 0 or h <= 0:
                    return None
                return Image(img_path, width=w, height=h)
            except Exception:
                return None

        story.append(PageBreak())
        story.append(Paragraph("图表详解", style_title))  # 居中黑体三号

        # 遍历 JSON：仅当 image_path 存在且本地可读时才插入图片 + 说明
        items = figure_analysis if isinstance(figure_analysis, list) else []
        available_width = A4[0] - (36 + 36)  # 与 SimpleDocTemplate 边距一致
        frame_max_w = available_width
        frame_max_h = (A4[1] - (36 + 36)) * 0.98  # 留一点余量，避免贴边报错

        for it in items:
            img_path = str(it.get("image_path", "")).strip()
            desc_md = str(it.get("rich_description_cn", "")).strip()
            if not img_path or not os.path.exists(img_path):
                # 跳过不存在的图片
                continue

            # ========= 图片 =========
            if img_path:
                img = make_scaled_image(img_path, frame_max_w, frame_max_h)
                # 如果你还在用“仅限宽版本”的 make_scaled_image，请改为：
                # img = make_scaled_image(real_img_path, frame_max_w)
                if img:
                    img.hAlign = 'CENTER'
                    story.append(img)
                    story.append(Spacer(1, 6))

            # ========= 说明（严格 Markdown + 段落分割 + 公式保护）=========
            if desc_md:
                for para in _split_paragraphs(desc_md):
                    story.append(Paragraph(md_to_rl_strict(para), style_md_body))
                story.append(Spacer(1, 12))

        # ====== Fuzzy 匹配与可视化工具（第三部分用） ======
        import string
        from difflib import SequenceMatcher

        def _normalize_space(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        def _clean_for_match(s: str) -> str:
            # 轻度清洗：去 Markdown 标记与标点、压缩空白、转小写
            t = md_to_plain(s)
            t = _normalize_space(t).lower()
            table = str.maketrans({c: " " for c in string.punctuation})
            t = t.translate(table)
            return _normalize_space(t)

        def _words_on_page(page):
            """
            返回按阅读顺序排列的单词信息:
            [(x0, y0, x1, y1, word), ...]
            """
            words = page.get_text("words")  # x0,y0,x1,y1, "word", block, line, word_no
            words.sort(key=lambda w: (w[5], w[6], w[7]))
            return [(w[0], w[1], w[2], w[3], w[4]) for w in words]

        def _page_text_blocks(page):
            """返回当前页的文本块矩形列表，用于避免把 [n] 画在文字上。"""
            blks = []
            for b in page.get_text("blocks") or []:
                try:
                    x0, y0, x1, y1 = b[:4]
                    blks.append(fitz.Rect(x0, y0, x1, y1))
                except Exception:
                    pass
            return blks
        
        def _fuzzy_find_window_rects(page, target_plain: str, min_len=3, threshold=0.82):
            """
            词窗模糊匹配：把 page 的 words 连接成窗口，与 target 做 difflib 相似度比较。
            命中后返回该窗口所有词的并集矩形列表（可进一步合并）。
            """
            ws = _words_on_page(page)
            if not ws:
                return []

            tgt = _clean_for_match(target_plain)
            if not tgt:
                return []
            tgt_tokens = tgt.split()
            L = max(min_len, min(len(tgt_tokens), 40))

            page_tokens = [_normalize_space(w[4].lower()) for w in ws]

            def _is_good(tok):
                return tok and any(ch.isalnum() for ch in tok)

            idx_map = [i for i, tok in enumerate(page_tokens) if _is_good(tok)]
            toks = [page_tokens[i] for i in idx_map]

            best = []
            scanned = 0
            MAX_WINDOWS = 1500
            tgt_joined = " ".join(tgt_tokens)

            for win_len in range(max(min_len, L - 5), L + 6):
                if win_len <= 0:
                    continue
                for start in range(0, max(0, len(toks) - win_len + 1)):
                    end = start + win_len
                    win_str = " ".join(toks[start:end])
                    ratio = SequenceMatcher(a=win_str, b=tgt_joined).ratio()
                    if ratio >= threshold:
                        word_idxs = idx_map[start:end]
                        rects = [fitz.Rect(ws[i][0], ws[i][1], ws[i][2], ws[i][3]) for i in word_idxs]
                        if rects:
                            union = rects[0]
                            for r in rects[1:]:
                                union |= r
                            best.append(union)
                    scanned += 1
                    if scanned >= MAX_WINDOWS:
                        break
                if best or scanned >= MAX_WINDOWS:
                    break
            return best

        def _find_label_spot_rightonly(page: fitz.Page, anchor_rect: fitz.Rect,
                                    margin=3, h=12, txt="[0]") -> fitz.Rect:
            """
            只在 anchor 右侧寻找可以放置 [n] 的位置；若与正文相交，则沿垂直方向小步扫描寻找空白。
            找不到空白则贴在右侧默认位置（不加底，不画框）。
            """
            # 宽度按字数自适应，避免被裁切
            base_w = 22 + 6 * max(0, len(txt) - 3)
            w = base_w
            page_rect = page.rect
            text_blocks = _page_text_blocks(page)

            # 先尝试：紧贴右侧，同一 y
            x0 = min(anchor_rect.x1 + margin, page_rect.x1 - w - margin)
            y0 = min(max(anchor_rect.y0, page_rect.y0 + margin), page_rect.y1 - h - margin)
            cand = fitz.Rect(x0, y0, x0 + w, y0 + h)
            if not _rect_intersects_any(cand, text_blocks):
                return cand

            # 垂直方向上下“阶梯”扫描，尽量找一个不与文字块相交的位置
            step = h + 2
            for k in range(1, 8):  # 扫描 7 个台阶，足够保守
                for dy in (k * step, -k * step):
                    yy = min(max(y0 + dy, page_rect.y0 + margin), page_rect.y1 - h - margin)
                    cand2 = fitz.Rect(x0, yy, x0 + w, yy + h)
                    if not _rect_intersects_any(cand2, text_blocks):
                        return cand2

            # 实在找不到空白，就用默认位置（可能与正文有些重叠，但满足“无白底/无边框”的要求）
            return cand

        def _rect_intersects_any(r: fitz.Rect, rects: list[fitz.Rect]) -> bool:
            for x in rects:
                if r.intersects(x):
                    return True
            return False

        def highlight_and_label(page, rects, gid, color=(1, 1, 0)):
            """
            高亮 rects，并在其右侧标注纯文本 [gid]。
            """
            if not rects:
                return

            # 高亮
            for r in rects:
                try:
                    page.add_highlight_annot(r)
                except Exception:
                    try:
                        page.draw_rect(r, color=color, fill=(1, 1, 0), fill_opacity=0.25, width=0)
                    except Exception:
                        pass

            # 右侧放置 [n]
            anchor = rects[0]
            label_text = f"[{gid}]"
            label_rect = _find_label_spot_rightonly(page, anchor, txt=label_text)
            label_point = fitz.Point(label_rect.x0, label_rect.y0)

            print("label rect:", label_rect)
            # 尝试写入，如果放不下就加宽/降字号重试
            try:
                # ret = page.insert_textbox(label_rect, label_text,
                #                         fontsize=10, fontname="helv",
                #                         color=(1, 0, 0), align=1, overlay=True)
                ret = page.insert_text(label_point, label_text,
                                       fontsize=10, fontname="helv",
                                        color=(0, 0, 1))
                # if ret == 0:  # 没写进去
                #     # 加宽 8pt 再试
                #     label_rect.x1 += 8
                #     ret = page.insert_textbox(label_rect, label_text,
                #                             fontsize=10, fontname="helv",
                #                             color=(1, 0, 0), align=1, overlay=True)
                # if ret == 0:  # 还不行就降字号
                #     ret = page.insert_textbox(label_rect, label_text,
                #                             fontsize=9, fontname="helv",
                #                             color=(1, 0, 0), align=1, overlay=True)
                print(f'insert_textbox ret={ret}')
            except Exception as e:
                print("insert_textbox failed:", e)

        # ====== 第三部分：直接在原文 PDF 上高亮并标号，然后与前两部分拼接 ======
        annotated_pdf_path = str((UPLOAD_DIR / f"{base_name}_annotated.pdf").resolve())
        src_doc = fitz.open(saved_pdf_path)

        # 构造页级快速过滤所需的低清洗文本
        pages_clean_text = []
        for _p in src_doc:
            try:
                t = _p.get_text("text")
            except Exception:
                t = ""
            t = re.sub(r"[ \t]+", " ", t.lower())
            pages_clean_text.append(t)

        # 候选片段：优先句首，其次前120/80/50字，增加命中率
        def candidate_snippets_from_md(md_src: str):
            t = md_to_plain(md_src)
            t = re.sub(r"\s+", " ", t).strip()
            if not t:
                return []
            sent = re.split(r"[。．.!?？]\s*", t)[0].strip() or t
            out = []
            if sent:
                out.append(sent[:120])
            out += [t[:120], t[:80], t[:50]]
            u = []
            for s in out:
                s = s.strip()
                if s and s not in u:
                    u.append(s)
            return u

        # 遍历全局 gid，逐个在原 PDF 中高亮 + 放置 [gid]
        gid_to_page = {}
        for gid, key, raw_md, plain_txt in ordered_by_gid:
            target_text = md_to_plain(raw_md)
            tgt_clean = re.sub(r"[ \t]+", " ", _clean_for_match(target_text))
            tgt_tokens = [w for w in tgt_clean.split() if any(ch.isalnum() for ch in w)]
            key_tokens = tgt_tokens[:8] if tgt_tokens else []

            for page_idx, page in enumerate(src_doc):
                rects = []

                # 页面级快速过滤：至少命中2个关键词
                if key_tokens:
                    page_txt = pages_clean_text[page_idx]
                    if sum(1 for w in key_tokens if w in page_txt) < 2:
                        continue

                # 1) 精确查找（多候选）
                for snip in candidate_snippets_from_md(raw_md):
                    try:
                        found = page.search_for(snip)
                    except Exception:
                        found = []
                    if found:
                        rects = found
                        break

                # 2) 失败则模糊匹配（对断行/表格更鲁棒）
                if not rects:
                    rects = _fuzzy_find_window_rects(page, target_text, min_len=3, threshold=0.82)

                if not rects:
                    continue

                highlight_and_label(page, rects, gid)
                gid_to_page[gid] = page.number  # 0-based
                break  # 每个 gid 命中一页即可

        # 保存带注释的原文 PDF
        try:
            src_doc.save(annotated_pdf_path, incremental=False, deflate=True)
        except Exception:
            try:
                src_doc.save(annotated_pdf_path)
            except Exception:
                annotated_pdf_path = saved_pdf_path
        finally:
            src_doc.close()

        # ====== 输出前两部分（分析报告部分） ======
        analysis_only_path = str((UPLOAD_DIR / f"{base_name}_analysis_only.pdf").resolve())
        doc = SimpleDocTemplate(
            analysis_only_path, pagesize=A4,
            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
        )
        doc.build(story)

        # ====== 拼接“分析报告部分 + 带注释原文” ======
        merger = PdfMerger()
        merger.append(analysis_only_path)
        merger.append(annotated_pdf_path)   # 直接把原 PDF 拼接上去
        merger.write(out_pdf_path)
        merger.close()

        return out_pdf_path

    except Exception as e:
        print(f"处理PDF时发生错误: {str(e)}")
        error_pdf_path = tempfile.NamedTemporaryFile(suffix='_error.pdf', delete=False).name
        try:
            c = canvas.Canvas(error_pdf_path, pagesize=A4)
            c.setFont("Helvetica", 12)
            c.drawString(72, A4[1] - 72, "生成分析PDF出错：")
            c.setFont("Helvetica", 10)
            c.drawString(72, A4[1] - 92, str(e))
            c.showPage()
            c.save()
        except Exception:
            pass
        return error_pdf_path

##################################################################
# TODO: 精读分析 - 添加后端对话API调用函数
def call_backend_chat_api(question: str) -> Dict[str, Any]:
    """
    调用后端对话API，返回答案和参考段落
    """
    url = "http://localhost:8004/process"

    data = {
        "text": question,  # 替换为你要分析的文本
    }

    response = requests.post(url, data=data)
    print(response)
    print(response.json())
    # time.sleep(5)

    # 临时模拟返回数据
    return response.json()

# 添加处理标注点击的函数
def handle_annotation_click(annotation_data):
    """
    处理标注点击事件，返回跳转信息
    """
    global ANNOTATION_MAP

    try:
        annotation_number = int(annotation_data)
        if annotation_number in ANNOTATION_MAP:
            annotation_info = ANNOTATION_MAP[annotation_number]
            return {
                'page': annotation_info['page'],
                'annotation_number': annotation_number,
                'action': 'jump'
            }
    except:
        pass

    return {'page': 0, 'action': 'none'}

def _add_user_and_thinking(user_input: str, messages: list | None):
    """把用户提问 + 一个占位的 assistant 'Thinking…' 加进 messages 列表（messages 模式）。"""
    msgs = list(messages or [])
    # 用户消息
    msgs.append({"role": "user", "content": (user_input or "")})
    # 占位的回答
    msgs.append({"role": "assistant", "content": "Thinking…"})
    return msgs

def _replace_last_assistant(msgs: list, new_content: str):
    """把最后一条 assistant 消息替换为真正的回答；若不存在则直接 append。"""
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], dict) and msgs[i].get("role") == "assistant":
            msgs[i] = {"role": "assistant", "content": new_content}
            return msgs
    msgs.append({"role": "assistant", "content": new_content})
    return msgs

def _render_refs_html(refs: list[str], expanded: bool, topk: int = 4) -> str:
    shown = refs if expanded else refs[:topk]
    items = []
    for idx, t in enumerate(shown, 1):
        esc = html.escape(t)
        cls = "analysis-ref-item full" if expanded else "analysis-ref-item"
        items.append(f'<div class="{cls}"><span class="secondary-text">[{idx}]</span> {esc}</div>')
    total = len(refs)
    title = f"<div class='analysis-ref-title'>Analysis Reference ({total})</div>"
    return f"<div class='analysis-ref-box'>{title}{''.join(items)}</div>"

def _toggle_refs(expanded_now: bool, refs: list[str]):
    new_expanded = not bool(expanded_now)
    if not refs:
        # 没有引用时，隐藏一切
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), new_expanded
    html_out = _render_refs_html(refs, new_expanded)
    btn_lab  = "Collapse ▲" if new_expanded else "Expand ▾"
    return gr.update(value="### Analysis Reference", visible=True), \
        gr.update(value=html_out, visible=True), \
        gr.update(value=btn_lab, visible=True), \
        new_expanded

# 实现analyze_paper函数
def analyze_paper(question: str, chat_history, pdf_files):
    """
    处理用户提问，调用后端API并生成带标注的回答
    """
    global ANNOTATION_MAP
    import html as _html

    msgs = list(chat_history or [])

    # 若不是通过事件链调用，兜底保证有一个占位 assistant
    def _ensure_placeholder(ms):
        if not ms or not (isinstance(ms[-1], dict) and ms[-1].get("role") == "assistant"):
            ms.append({"role": "assistant", "content": "Thinking…"})
        return ms

    # 基本校验
    if not pdf_files:
        msgs = _replace_last_assistant(_ensure_placeholder(msgs), "请先上传 PDF 文件再提问。")
        return (msgs, gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), [], False)

    if not question or not question.strip():
        msgs = _replace_last_assistant(_ensure_placeholder(msgs), "请输入有效的问题。")
        return (msgs, gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), [], False)

    try:
        # 选第一份“原始上传的 PDF”
        orig_pdf_path = pdf_files[0] if isinstance(pdf_files, list) else pdf_files

        # === 调后端 ===
        result = call_backend_chat_api(question)
        answer = (result.get("response") or "").strip()
        refs   = result.get("reference") or []
        ref_pages = result.get("reference_pages") or [1] * len(refs)
        if len(ref_pages) != len(refs):
            ref_pages = [1] * len(refs)

        # === 构建可点击引用 + 映射 ===
        ANNOTATION_MAP.clear()
        marks = []
        for i, (ref_text, page) in enumerate(zip(refs, ref_pages), 1):
            preview = " ".join(ref_text.split()[:32]) + ("..." if len(ref_text.split()) > 32 else "")
            ANNOTATION_MAP[i] = {"text": ref_text, "page": max(0, int(page) - 1), "file": orig_pdf_path, "preview": preview}
            # 仅 title，去掉 onclick / data-ref
            marks.append(f'<span class="annotation" title="{_html.escape(preview)}">[{i}]</span>')

        html_answer = answer + ((" " + " ".join(marks)) if marks else "")

        # === 只替换最后一条 assistant（即占位 Thinking…）===
        msgs = _replace_last_assistant(_ensure_placeholder(msgs), html_answer)

        # refs 可能为空：为空则隐藏参考框也不拼 [n]
        if not refs:
            html_answer = answer  # 不拼 [n]
            msgs = _replace_last_assistant(_ensure_placeholder(msgs), html_answer)
            return (msgs,
                    gr.update(visible=False),  # ref_title
                    gr.update(visible=False),  # ref_html
                    gr.update(visible=False),  # ref_toggle_btn
                    [],                        # refs_state
                    False)                     # refs_expanded
        else:
            # 上面“只带 title 的 [n]”代码……
            html_answer = answer + " " + " ".join(marks)
            msgs = _replace_last_assistant(_ensure_placeholder(msgs), html_answer)
            # 初始：显示最多4条、折叠态
            return (msgs,
                    gr.update(value="### Analysis Reference", visible=True),
                    gr.update(value=_render_refs_html(refs, expanded=False), visible=True),
                    gr.update(value="Expand ▾", visible=True),
                    refs,
                    False)

    except Exception as e:
        logger.error(f"处理问题失败: {e}")
        msgs = _replace_last_assistant(_ensure_placeholder(msgs), f"处理问题时发生错误: {str(e)}")
        return (msgs, gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), [], False)

# 添加PDF跳转函数
def jump_to_annotation(pdf_viewer, annotation_info):
    """
    跳转到指定标注的页面
    """
    if annotation_info.get('highlight', False):
        page_num = annotation_info.get('page', 0)
        annotation_number = annotation_info.get('annotation_number')

        # 这里需要根据具体的PDF查看器API来实现跳转
        # 返回更新后的PDF查看器状态
        return {
            'page': page_num,
            'highlight_annotation': annotation_number
        }

    return pdf_viewer

##################################################################
# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas { max-width: 100%; height: auto; image-rendering: auto; }

    /* 聊天界面样式 */
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .input-container {
        display: flex;
        gap: 10px;
        align-items: center;
    }

    .submit-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    /* 标注样式 */
    .annotation {
        color: #165DFF;
        cursor: default;         /* 不再是 pointer */
        font-weight: 600;
        padding: 0;
        border: none;
        background: transparent;
        margin: 0 2px;
    }
    .annotation:hover { background: transparent; border: none; }

    /* 工具提示样式 */
    .tooltip {
        position: absolute;
        background: white;
        border: 1px solid #ccc;
        padding: 8px;
        border-radius: 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 420px;          /* 控制单行过长 */
        font-size: 12px;

        white-space: normal;       /* 允许换行 */
        word-break: break-word;    /* 长单词也断行 */
        overflow-wrap: anywhere;   /* 任何位置都可断行 */

        display: -webkit-box;      /* 下面三行做“5行截断” */
        -webkit-line-clamp: 5;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .left-col .gr-markdown h2 { margin: 6px 0 !important; }
    .left-col .gr-block { margin-top: 8px !important; }
    .left-col .gr-form, .left-col .gr-group { gap: 8px !important; }
    .right-col > .gr-block { margin-top: 6px !important; }
    .right-col .gr-row, .right-col .gr-group, .right-col .gr-form { gap: 6px !important; }
    .right-col .gr-markdown { margin: 4px 0 !important; }
    .right-col .gr-file { border-width: 1px !important; box-shadow: none !important; }
    .right-col .gr-button { margin: 6px 0 !important; min-height: auto !important; }
    .right-col .gr-button > button { height: auto !important; padding: 10px 14px !important; }
    
    .gr-tabs [role="tablist"],
    .gr-tabs .tab-nav,
    .gr-tabs > div:first-child { display: none !important; }
    
    .analysis-ref-box { border:1px solid #e0e0e0; border-radius:8px; padding:12px; margin-top:8px; }
    .analysis-ref-title { font-weight:600; margin-bottom:8px; }
    .analysis-ref-item { margin:6px 0; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
    .analysis-ref-item.full { display:block; -webkit-line-clamp:unset; }
"""

# 添加JavaScript代码处理标注交互
js_code = """
<script>
// 全局变量存储标注信息
let annotationData = {};

// 显示标注提示
function showAnnotationTooltip(refNumber) {
    const annotation = document.querySelector(`[data-ref="${refNumber}"]`);
    if (annotation && annotationData[refNumber]) {
        // 移除现有的工具提示
        hideAnnotationTooltip();

        // 创建新的工具提示
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.innerHTML = annotationData[refNumber];
        tooltip.id = 'annotation-tooltip';

        // 定位工具提示
        const rect = annotation.getBoundingClientRect();
        tooltip.style.position = 'fixed';
        tooltip.style.left = (rect.left + window.scrollX) + 'px';
        tooltip.style.top = (rect.top + window.scrollY - 40) + 'px';

        document.body.appendChild(tooltip);
    }
}

// 隐藏标注提示
function hideAnnotationTooltip() {
    const tooltip = document.getElementById('annotation-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// 监听Gradio返回的标注数据
document.addEventListener('annotation-data', function(e) {
    annotationData = e.detail;
});

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    // 监听标注点击事件
    document.addEventListener('annotation-click', function(e) {
        // 这里可以通过Gradio的Python函数处理点击事件
        console.log('Annotation clicked:', e.detail.refNumber);
    });
});
</script>
"""

tech_details_string = f"""
                    <summary>Technical details</summary>
                    - Tool Name: DigPaper——基于RAG的论文精读Agent<br>
                    - Version: 1.0.0 <br>
                    - Author: 爱吃肥肥<br>
                    - Supported Formats: PDF Documents<br>
                """
cancellation_event_map = {}

# The following code creates the GUI
with gr.Blocks(
        title="DigPaper",
        theme=gr.themes.Default(primary_hue=custom_blue, spacing_size="md", radius_size="lg"),
        css=custom_css,
) as demo:
    # 添加JavaScript代码
    gr.HTML(js_code)

    # ===================== 顶部标题（左上角项目名称） =====================
    gr.Markdown("# DigPaper")

    # ===================== 主体 =====================
    with gr.Row(equal_height=False):
        # --------------------- 左列 ---------------------
        with gr.Column(scale=1, elem_classes=["left-col"]):
            # 上传文件
            gr.Markdown("## File")
            file_input = gr.File(
                label=None,  # ← 改成 None 或 show_label=False
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["input-file"],
            )

            # 文件预览
            gr.Markdown("## Preview")
            pdf_list_state = gr.State([])
            pdf_selector = gr.Dropdown(
                label=None, choices=[], value=None,
                interactive=True, visible=False, show_label=False
            )
            pdf_view = PDF(label=None, height=1200, interactive=True, visible=False, elem_id="preview_pdf")

            # 添加PDF查看器状态
            pdf_viewer_state = gr.State({"page": 0, "highlight_annotation": None})

            # 版权信息
            gr.Markdown(
                f"<div style='text-align:center;color:#888;font-size:12px;'>© {datetime.now().year} DigPaper. All rights reserved.</div>",
                elem_id="copyright")

        # --------------------- 右列 ---------------------
        with gr.Column(scale=1, elem_classes=["right-col"]):
            # 全文分析区
            gr.Markdown("## Full-text Analysis")
            F_analysis_btn = gr.Button("✨ Full-text Analysis", variant="primary")
            F_analysis_output_title = gr.Markdown("**Analysis Completed: Download Generated PDF**", visible=False)
            F_analysis_output_file = gr.File(label="Download Link (Click to Download)", visible=False)

            analysis_pdf_state = gr.State("")

            # 精读分析区
            gr.Markdown("## Chat With Paper")

            # 聊天对话容器
            with gr.Column(elem_classes=["chat-container"]):
                chatbot = gr.Chatbot(
                    label="chatbot",
                    height=380,
                    show_copy_button=True,
                    type="messages",
                    render_markdown=True,
                    show_label=False
                )

                with gr.Row(elem_classes=["input-container"]):
                    user_input = gr.Textbox(
                        placeholder="Enter your question here",
                        lines=1,
                        scale=8,
                        show_label=False,
                    )
                    submit_btn = gr.Button("▶ Submit", variant="primary", scale=1, elem_classes=["submit-btn"])

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear")
            
            # ===== Analysis Reference UI =====
            refs_state = gr.State([])          # 保存所有参考段落
            refs_expanded = gr.State(False)    # 展开/收起状态
            ref_title = gr.Markdown(visible=False)
            ref_html  = gr.HTML(visible=False)
            ref_toggle_btn = gr.Button("Expand ▾", visible=False)

            # 标注交互组件
            annotation_tooltip_data = gr.Textbox(visible=False)
            pdf_jump_data = gr.JSON(value={"page": 0, "action": "none"}, visible=False)
    ##################################################################
    def register_analysis_pdf(analysis_path, pdf_paths):
        if not analysis_path:
            return (
                gr.update(visible=False),  # F_analysis_output_title
                gr.update(visible=False),  # F_analysis_output_file
                gr.update(),  # pdf_view
                pdf_paths,  # pdf_list_state
                gr.update(),  # pdf_selector
            )

        new_list = list(pdf_paths or [])
        if analysis_path not in new_list:
            new_list.append(analysis_path)

        choices = [(os.path.basename(p), p) for p in new_list]  # label=名, value=路径

        title_update = gr.update(value="**Analysis Completed: Download Generated PDF**", visible=True)
        file_update = gr.update(value=analysis_path, visible=True)
        view_update = gr.update(value=analysis_path, visible=True)
        selector_update = gr.update(choices=choices, value=analysis_path, visible=True)

        return title_update, file_update, view_update, new_list, selector_update

    def update_pdf_selector(paths):
        names = [os.path.basename(p) for p in (paths or [])]
        default_val = (names[-1] if names else None)
        return gr.update(choices=names, value=default_val, visible=bool(names))

    def select_pdf(selected_path):
        if not selected_path:
            return gr.update(visible=False)
        return gr.update(value=selected_path, visible=True)

    # 添加处理上传文件的函数
    def handle_uploaded_files(files):
        files = files or []
        paths = []
        for f in files:
            p = f.name if hasattr(f, "name") else f
            paths.append(save_uploaded_file(p))

        if not paths:
            return (
                gr.update(choices=[], value=None),  # pdf_selector
                gr.update(value=None),              # pdf_view
                [],                                 # pdf_list_state
                gr.update(visible=False),           # ref_title
                gr.update(visible=False),           # ref_html
                gr.update(visible=False),           # ref_toggle_btn
                [],                                 # refs_state
                False,                              # refs_expanded
            )

        latest = paths[-1]
        choices = [(os.path.basename(p), p) for p in paths]
        return (
            gr.update(choices=choices, value=latest, visible=True),
            gr.update(value=latest, visible=True),
            paths,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            False,
        )


    # 添加处理提交按钮状态的函数
    def toggle_submit_button():
        """禁用提交按钮"""
        return gr.Button(interactive=False)

    def enable_submit_button():
        """启用提交按钮"""
        return gr.Button(interactive=True)

    def after_generate(prev_list, new_pdf_path):
        new_list = list(prev_list or [])
        if new_pdf_path and new_pdf_path not in new_list:
            new_list.append(new_pdf_path)

        names = [os.path.basename(p) for p in new_list]
        return (
            gr.update(choices=names, value=os.path.basename(new_pdf_path), visible=True),  # selector
            gr.update(value=new_pdf_path, visible=True),  # 预览
            new_list  # state(完整路径)
        )

    # 绑定事件处理
    file_input.upload(
        fn=handle_uploaded_files,
        inputs=[file_input],
        outputs=[pdf_selector, pdf_view, pdf_list_state, ref_title, ref_html, ref_toggle_btn, refs_state, refs_expanded]
    )

    def _disable_chat_controls():
        return gr.update(interactive=False), gr.update(interactive=False)

    def _enable_chat_controls():
        return gr.update(interactive=True), gr.update(interactive=True)
    
    # 点按钮先禁用自己（立刻执行，不进队列）
    F_analysis_btn.click(
        fn=lambda: gr.update(value="Processing…", interactive=False),
        outputs=[F_analysis_btn],
        queue=False
        # 主任务：用“下拉选择的完整路径”作为输入
    ).then(
        fn=F_analyze_paper,
        inputs=[pdf_selector],  # ← 改这里！用 dropdown 的 value（完整路径）
        outputs=[analysis_pdf_state],
        queue=True
        # 注册结果（展示下载、预览、把新 PDF 加到列表中）
    ).then(
        fn=register_analysis_pdf,
        inputs=[analysis_pdf_state, pdf_list_state],
        outputs=[F_analysis_output_title, F_analysis_output_file, pdf_view, pdf_list_state, pdf_selector],
        queue=False
        # 处理完恢复按钮状态
    ).then(
        fn=lambda: gr.update(value="✨ Full-text Analysis", interactive=True),
        outputs=[F_analysis_btn],
        queue=False
    )

    pdf_selector.change(
        fn=select_pdf,
        inputs=[pdf_selector],  # 现在 dropdown 的 value 就是“完整路径”
        outputs=[pdf_view],
    )

    # 提交问题
    submit_btn.click(
        fn=_disable_chat_controls, outputs=[submit_btn, clear_btn], queue=False
    ).then(
        fn=_add_user_and_thinking,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
        queue=False
    ).then(
        fn=analyze_paper,
        inputs=[user_input, chatbot, pdf_list_state],
        outputs=[chatbot, ref_title, ref_html, ref_toggle_btn, refs_state, refs_expanded],
        queue=True
    ).then(
        fn=_enable_chat_controls, outputs=[submit_btn, clear_btn], queue=False
    ).then(
        fn=lambda: "", outputs=[user_input], queue=False
    )

    user_input.submit(
        fn=_disable_chat_controls, outputs=[submit_btn, clear_btn], queue=False
    ).then(
        fn=_add_user_and_thinking,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
        queue=False
    ).then(
        fn=analyze_paper,
        inputs=[user_input, chatbot, pdf_list_state],
        outputs=[chatbot, ref_title, ref_html, ref_toggle_btn, refs_state, refs_expanded],
        queue=True
    ).then(
        fn=_enable_chat_controls, outputs=[submit_btn, clear_btn], queue=False
    ).then(
        fn=lambda: "", outputs=[user_input], queue=False
    )

    ref_toggle_btn.click(
        fn=_toggle_refs,
        inputs=[refs_expanded, refs_state],
        outputs=[ref_title, ref_html, ref_toggle_btn, refs_expanded],
    )
    def _clear_chat_and_refs():
        # 清空聊天 + 隐藏参考框 + 重置 state
        return ([],
                gr.update(visible=False),  # ref_title
                gr.update(visible=False),  # ref_html
                gr.update(visible=False),  # ref_toggle_btn
                [],                        # refs_state
                False)                     # refs_expanded

    # 清空聊天记录
    clear_btn.click(
        fn=_clear_chat_and_refs,
        outputs=[chatbot, ref_title, ref_html, ref_toggle_btn, refs_state, refs_expanded]
    )

##################################################################

def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
        share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)
    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
    else:
        if len(user_list) == 0:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True, inbrowser=True, share=True, server_port=server_port
                    )
        else:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True,
                        inbrowser=True,
                        share=True,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                        # , delete_cache=True,
                    )

### 新的做了安全控制的GUI
def setup_gui_new(
        user_list, 
        debug=True, 
        server_port=7860,
        max_file_size = "5mb",
) -> None:
    
    if debug:
        inbrower = True
        auth_message = 'DigPaper调试中，请登录。'
        server_ip = '127.0.0.1'
        
    else:
        inbrower = False
        auth_message = '欢迎使用DigPaper!请登录。'
        server_ip = '0.0.0.0'
        

    if len(user_list) == 0:
        raise ValueError("No user name and password found.")
    else:
        try:
            demo.launch(
                server_name=server_ip,
                debug=True,
                inbrowser=inbrower,
                share=False,
                auth=user_list,
                auth_message=auth_message,
                max_file_size=max_file_size,
                server_port=server_port,
                delete_cache=True,
            )
        except Exception:
            print("Failed to launch GUI.")


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    user_list = [
    ]
    setup_gui_new(user_list, debug=False)
