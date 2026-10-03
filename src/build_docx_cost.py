# -*- coding: utf-8 -*-
"""Dựng DOCX chuẩn IEEE Access cho bài cost-aware và tài liệu bổ sung.

    paper/draft_cost_v3.md          -> paper/GA4XGB_cost_v3_IEEE_Access.docx
    paper/supplementary_cost_v2.md  -> paper/GA4XGB_cost_v3_Supplementary.docx

Vì sao một builder mới thay vì sửa src/build_docx.py: builder cũ gắn chặt với bài GA
(tên hình, chú thích hình viết cứng trong mã, abstract nhận diện bằng câu mở đầu, chỉ
hai phương trình dựng tay). Bài cost-aware viết chú thích ngay trong markdown, có bảng
nhiều ô (a)/(b), Algorithm 1 và sáu phương trình. Sửa builder cũ sẽ làm hỏng khả năng
dựng lại bản GA đã nộp. Luật tạp chí vẫn lấy từ src/docx_ieee.py, toán từ src/mathrun.py.

Quy trình một chiều như trước: sửa markdown rồi chạy lại, không sửa tay DOCX.

Quy ước markdown mà parser này đọc (đúng như draft_cost_v3.md đang viết):
  * "# " tiêu đề; dòng tác giả và affiliation dùng <sup>n</sup>.
  * "## ABSTRACT" + một đoạn; dòng bắt đầu "INDEX TERMS".
  * "## I. X" mục cấp 1 (số La Mã gõ tay), "## APPENDIX A. X", "## DATA AVAILABILITY"…
    là mục không số; "### A. X" mục cấp 2.
  * "TABLE n. …" đứng TRÊN bảng; có thể có các nhãn "(a)", "(b)" rồi bảng; đoạn
    "Note. …" ngay sau là chú thích cuối bảng.
  * "![…](figures_cost/x.png)" rồi dòng "FIGURE n. …" là chú thích hình (đặt DƯỚI hình).
  * "$$…$$" một dòng: phương trình trưng bày, dịch bằng mathrun.display_omml.
  * Khối ``` … ``` bắt đầu bằng "Algorithm": hộp thuật toán.
  * Toán trong dòng viết dạng chữ thường (cost_K, μ_K^G, μ_w^{σ(X)}): đổi thành chỉ số
    dưới/trên thật bằng math_tokens(); định danh như rs_tuned, kings_county giữ nguyên.
  * [n] thành liên kết nội bộ tới mục tài liệu n; [TBD…] tô vàng để tác giả thấy.

Chạy: PYTHONPATH=src python3 src/build_docx_cost.py           (dựng cả hai + kiểm)
      PYTHONPATH=src python3 src/build_docx_cost.py --check   (chỉ kiểm hai file đã dựng)
"""
import argparse
import os
import re
import sys
import zipfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_ieee import (add_bookmark, add_cite_link, clear_body, format_table_ieee,  # noqa: E402
                       no_autonum, ref_anchor, repeat_header, set_columns, set_indent,
                       shrink_break, _border)
from mathrun import W_NS, break_long, display_omml, split_display  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(REPO, "paper", "Access-Template-2024.docx")
JOBS = {
    "main": (os.path.join(REPO, "paper", "draft_cost_v3.md"),
             os.path.join(REPO, "paper", "GA4XGB_cost_v3_IEEE_Access.docx")),
    "supp": (os.path.join(REPO, "paper", "supplementary_cost_v2.md"),
             os.path.join(REPO, "paper", "GA4XGB_cost_v3_Supplementary.docx")),
}

# Bề rộng khả dụng của template: khổ 8 in, lề 0,514 in -> 6,97 in; cột = (6,97 - 0,28)/2.
# Hướng dẫn chung của IEEE ghi 7,16 in và 3,5 in, nhưng đó là khổ 8,5 in của bản in;
# template Access 2024 nhỏ hơn, đặt 7,16 in vào đây sẽ tràn lề. Lấy số của template.
FULL_IN, COL_IN = 6.95, 3.33
WIDE_FIG_MIN_IN = 4.0          # PNG rộng hơn (theo DPI ghi trong file) -> hình khổ đôi
TABLE_PT = 7.0
EM_IN = 0.15                   # 1 em của Cambria Math 10 pt (0,139 in) cộng biên an toàn, cho break_long
CHAR_IN = 0.047                # bề rộng trung bình một ký tự Times 7 pt, để đoán bảng hẹp/rộng


# ---------------------------------------------------------------------------
# Toán trong dòng viết dạng chữ: cost_K, μ_K^G, μ_w^{σ(X)}, p̂(y)^(−λ), x_1x_2
# ---------------------------------------------------------------------------
_COMBINING = "̂̄̅̃̇"
_SUBDIGIT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _is_letter(c):
    return c.isalpha() and c not in "_"


def _base_ok(s, i):
    """'_' ở vị trí i có phải chỉ số dưới toán không? Gốc phải là MỘT chữ cái (có thể
    kèm dấu kết hợp như ŷ, μ̂, w̄) đứng sau ranh giới từ, hoặc chữ 'cost'. Nhờ vậy
    rs_tuned, kings_county, max_depth, R8_bag5, MAT_S11 giữ nguyên."""
    j = i - 1
    while j >= 0 and s[j] in _COMBINING:
        j -= 1
    if j < 0 or not _is_letter(s[j]):
        return False
    if s[j - 4 + 1:j + 1] == "cost" and (j - 4 < 0 or not (s[j - 4].isalnum() or s[j - 4] == "_")):
        return True
    if j == 0 or s[j - 1].isdigit() and (j < 2 or not s[j - 2].isalpha()):
        return True                    # 2p_T: hệ số trước biến (nhưng không phải S11_x)
    if s[j - 1] == "d" and (j < 2 or not (s[j - 2].isalnum() or s[j - 2] == "_")):
        return True                    # dr_T, dr_M: vi phân
    return not (s[j - 1].isalnum() or s[j - 1] in "_.")


def _read_script(s, i):
    """Đọc đối số chỉ số bắt đầu tại s[i]; trả (văn bản, vị trí sau đối số) hoặc None."""
    if i >= len(s):
        return None
    c = s[i]
    if c == "{":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        inner = s[i + 1:j - 1]
        # chỉ số lồng (μ̂_w^{G_1}, r_{(τ_j, b)}): số dưới lồng thành ký tự ₁, chữ giữ nguyên
        inner = re.sub(r"_(\d+)", lambda m: m.group(1).translate(_SUBDIGIT), inner)
        inner = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: m.group(1) or m.group(2), inner)
        return inner, j
    if c == "(":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += (s[j] == "(") - (s[j] == ")")
            j += 1
        return s[i + 1:j - 1], j
    if c == "|":
        j = s.find("|", i + 1)
        if j > 0:
            return s[i:j + 1], j + 1
        return None
    if c.isdigit():
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        return s[i:j], j
    if c.isascii() and c.isalpha():
        j = i
        while j < len(s) and s[j].isascii() and s[j].isalnum():
            j += 1
        return s[i:j], j
    if c.isalpha() or c in "−-":                       # một chữ Hy Lạp, hoặc dấu trừ + chữ
        return c, i + 1
    return None


def math_tokens(s):
    """Cắt chuỗi thành [(text, level)], level ∈ {'', 'sub', 'sup'}."""
    out, buf, i, n = [], "", 0, len(s)
    while i < n:
        c = s[i]
        if c == "_" and _base_ok(s, i):
            got = _read_script(s, i + 1)
            # đối số toàn chữ hoa nhiều ký tự (G_SC, F_PRO…) là tên cột, không phải toán
            if got and not (len(got[0]) > 1 and re.fullmatch(r"[A-Z0-9]+", got[0])
                            and re.search(r"[A-Z]{2}", got[0])):
                arg, j = got
                # F_dt-cn-bc: tên tập đặc trưng, gạch nối thuộc chỉ số
                if s[i - 1] == "F" and arg in ("dt", "full"):
                    m = re.match(r"(?:-(?:cn|bc))+", s[j:])
                    if m:
                        arg, j = arg + m.group(0), j + m.end()
                if j < n and s[j] == "_":               # chuỗi định danh a_b_c: bỏ qua
                    buf += c
                    i += 1
                    continue
                if buf:
                    out.append((buf, ""))
                    buf = ""
                out.append((arg, "sub"))
                i = j
                continue
        if c == "^" and i > 0 and (s[i - 1].isalnum() or s[i - 1] in ")}]" + _COMBINING
                                   or out and out[-1][1] == "sub" and not buf):
            got = _read_script(s, i + 1)
            if got:
                arg, j = got
                if buf:
                    out.append((buf, ""))
                    buf = ""
                out.append((arg.replace("-", "−"), "sup"))
                i = j
                continue
        buf += c
        i += 1
    if buf:
        out.append((buf, ""))
    return out


# ---------------------------------------------------------------------------
# Run: nghiêng/đậm, <sup>, [TBD], [n], toán chữ
# ---------------------------------------------------------------------------
_ITAL = re.compile(r"(\*\*[^*]+\*\*|(?<![\w*])\*(?=\S)[^*]+?(?<=\S)\*(?![\w*]))")
_SPECIAL = re.compile(r"(<sup>.*?</sup>|\[TBD[^\]]*\]|\[\d+\])")


def _style_run(r, bold=False, italic=False, size=None, level=""):
    if bold:
        r.bold = True
    if italic:
        r.italic = True
    if size:
        r.font.size = Pt(size)
    if level == "sub":
        r.font.subscript = True
    elif level == "sup":
        r.font.superscript = True
    return r


def emit(par, text, link_cites=False, bold=False, italic=False, size=None, math=True):
    """Đổ một đoạn markdown vào `par` theo thứ tự xuất hiện.

    math=False cho mục tài liệu tham khảo: DOI như 10.1007/978-3-540-24775-3_3 có '_'
    thật, không được biến thành chỉ số dưới."""
    for chunk in _ITAL.split(text):
        if not chunk:
            continue
        b, it = bold, italic
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            chunk, b = chunk[2:-2], True
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2 and _ITAL.fullmatch(chunk):
            chunk, it = chunk[1:-1], True
        for piece in _SPECIAL.split(chunk):
            if not piece:
                continue
            if piece.startswith("<sup>"):
                _style_run(par.add_run(piece[5:-6]), b, it, size, "sup")
            elif piece.startswith("[TBD"):
                r = _style_run(par.add_run(piece), b, it, size)
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            elif re.fullmatch(r"\[\d+\]", piece) and link_cites:
                add_cite_link(par, piece, ref_anchor(piece[1:-1]))
                if size or b or it:                      # giữ cỡ chữ trong bảng/chú thích
                    rr = par._p[-1].find(qn("w:r"))
                    rPr = rr.get_or_add_rPr() if hasattr(rr, "get_or_add_rPr") else None
                    if rPr is None:
                        rPr = OxmlElement("w:rPr")
                        rr.insert(0, rPr)
                    if size:
                        sz = OxmlElement("w:sz")
                        sz.set(qn("w:val"), str(int(size * 2)))
                        rPr.append(sz)
            else:
                for txt, lvl in (math_tokens(piece) if math else [(piece, "")]):
                    _style_run(par.add_run(txt), b, it, size, lvl)
    return par


# ---------------------------------------------------------------------------
# Bảng
# ---------------------------------------------------------------------------
def parse_table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]


def _plain_len(cell):
    return len(re.sub(r"[*_^{}]", "", cell))


def col_widths(rows, total):
    """Bề rộng cột tỉ lệ với độ dài nội dung (chặn trên để cột chữ dài không nuốt hết)."""
    ncol = max(len(r) for r in rows)
    L = []
    for j in range(ncol):
        cells = [r[j] if j < len(r) else "" for r in rows]
        body = [_plain_len(c) for c in cells[1:]] or [0]
        head = max((len(w) for w in re.sub(r"[*_^{}]", "", cells[0]).split()), default=0) + 1
        longest = max(body)
        # cột chữ dài: dùng trung bình có trọng số để không chiếm cả bảng
        typical = sorted(body)[int(0.8 * (len(body) - 1))] if body else 0
        # từ dài nhất không ngắt được (medianHouseValue, tên file) phải nằm gọn một dòng
        word = max((len(w) for c in cells[1:] for w in re.sub(r"[*_^{}]", "", c).split()), default=0)
        L.append(max(min(max(typical, longest * 0.6), 64), min(head, 18), min(word, 30) * 1.05, 4))
    s = sum(L)
    return [total * x / s for x in L]


def natural_width(rows):
    ncol = max(len(r) for r in rows)
    w = 0.0
    for j in range(ncol):
        cells = [_plain_len(r[j]) if j < len(r) else 0 for r in rows]
        w += min(max(cells), 34) * CHAR_IN + 0.1
    return w


def add_table(doc, rows, total, link_cites, keep=True):
    """keep=True: bảng không gãy giữa hai cột/trang (bảng ngắn); bảng dài vẫn được
    tràn, với hàng tiêu đề lặp lại."""
    ncol = max(len(r) for r in rows)
    # cột chữ dài (lý do, mô tả) canh trái; cột số canh giữa
    textcol = [j == 0 or sum(_plain_len(r[j]) for r in rows[1:] if j < len(r))
               / max(len(rows) - 1, 1) > 18 for j in range(ncol)]
    t = doc.add_table(rows=len(rows), cols=ncol)
    keep = keep and len(rows) <= 22
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            txt = row[j] if j < len(row) else ""
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if textcol[j] else WD_ALIGN_PARAGRAPH.CENTER
            emit(p, txt, link_cites=link_cites, bold=(i == 0), size=TABLE_PT)
            if i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if not textcol[j] or j else WD_ALIGN_PARAGRAPH.LEFT
            if keep and i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            # dấu đoạn của ô mang cỡ chữ của style Normal (12 pt): ô trống làm hàng cao
            # gấp rưỡi hàng khác. Đặt cỡ dấu đoạn bằng cỡ chữ bảng.
            mark = OxmlElement("w:rPr")
            msz = OxmlElement("w:sz")
            msz.set(qn("w:val"), str(int(TABLE_PT * 2)))
            mark.append(msz)
            p._p.get_or_add_pPr().append(mark)
        trPr = t.rows[i]._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
    format_table_ieee(t, col_widths(rows, total), font_pt=TABLE_PT)
    repeat_header(t)
    return t


def add_algorithm(doc, lines, width):
    """Hộp thuật toán: bảng một cột, viền trên/dưới như luật IEEE, tiêu đề đậm có
    gạch dưới, các bước có thụt treo. Không dùng font đơn cách: dòng dài nhất của
    Algorithm 1 là 78 ký tự, ở Courier 7 pt rộng 4,5 in, tràn cột 3,33 in."""
    title, steps = lines[0].strip(), []
    for ln in lines[1:]:
        if ln.startswith((" ", "\t")) and steps:
            steps[-1] += " " + ln.strip()
        elif ln.strip():
            steps.append(ln.strip())
    t = doc.add_table(rows=2, cols=1)
    tbl = t._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblStyle")):
        tblPr.remove(old)
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(width * 1440)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)
    bd = OxmlElement("w:tblBorders")
    _border(bd, "top", "single", 12, "000000")
    _border(bd, "bottom", "single", 12, "000000")
    tblPr.append(bd)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    grid = tbl.find(qn("w:tblGrid"))
    for gc in grid.findall(qn("w:gridCol")):
        gc.set(qn("w:w"), str(int(width * 1440)))
    c0, c1 = t.cell(0, 0), t.cell(1, 0)
    # hộp không được gãy giữa hai cột/trang: hàng không tách, hàng tiêu đề đi cùng hàng thân
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
    for c in (c0, c1):
        c.width = Inches(width)
    tcPr = c0._tc.get_or_add_tcPr()
    tb = OxmlElement("w:tcBorders")
    _border(tb, "bottom", "single", 6, "000000")
    tcPr.append(tb)
    p = c0.paragraphs[0]
    p.paragraph_format.keep_with_next = True
    m = re.match(r"(Algorithm \d+\.)(.*)", title)
    _style_run(p.add_run(m.group(1) if m else title), bold=True, size=8)
    if m:
        emit(p, m.group(2), size=8)
    first = True
    for st in steps:
        p = c1.paragraphs[0] if first else c1.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_before = Pt(0.5)
        pf.space_after = Pt(0.5)
        pf.line_spacing = 1.0
        mnum = re.match(r"(\d+\.|Input:|Output:)\s*(.*)", st)
        if mnum:
            set_indent(p, left=300, hanging=300)
            _style_run(p.add_run(mnum.group(1) + "\t"), bold=mnum.group(1).endswith(":"), size=8)
            txt = mnum.group(2).replace(">=", "≥").replace("<=", "≤")
            txt = re.sub(r"\byhat_(\w+)", r"ŷ_\1", txt).replace("yhat", "ŷ")
            emit(p, txt, size=8)
        else:
            emit(p, st, size=8)
    return t


_TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
                "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
                "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]


# thứ tự con của w:pPr theo schema OOXML (CT_PPrBase); numPr đứng sau pStyle và các
# phần tử keepNext…widowControl, trước tabs/spacing/ind/jc
_PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
              "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
              "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN",
              "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing",
              "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment",
              "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]


def order_ppr(doc):
    """Sắp con của mọi w:pPr theo thứ tự schema. docx_ieee.set_indent nối w:ind vào
    cuối và shrink_break chèn w:rPr lên đầu; Word hiện bỏ qua, nhưng file đúng schema
    thì không phụ thuộc vào độ dễ dãi đó."""
    rank = {f"{{{W_NS}}}{n}": i for i, n in enumerate(_PPR_ORDER)}
    for pPr in doc.element.body.iter(qn("w:pPr")):
        kids = list(pPr)
        kids.sort(key=lambda e: rank.get(e.tag, 98))
        for e in kids:
            pPr.remove(e)
            pPr.append(e)


def order_tblpr(doc):
    """Sắp con của w:tblPr theo thứ tự schema. docx_ieee.format_table_ieee nối
    tblW/tblBorders/tblLayout SAU tblLook; Word hiện vẫn đọc được, nhưng trình
    kiểm schema và một số phiên bản Word coi là lỗi."""
    for tblPr in doc.element.body.iter(qn("w:tblPr")):
        kids = list(tblPr)
        rank = {f"{{{W_NS}}}{n}": i for i, n in enumerate(_TBLPR_ORDER)}
        kids.sort(key=lambda e: rank.get(e.tag, 99))
        for e in kids:
            tblPr.remove(e)
            tblPr.append(e)


# ---------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------
def fig_is_wide(png):
    from PIL import Image
    im = Image.open(png)
    dpi = (im.info.get("dpi") or (300, 300))[0] or 300
    return im.size[0] / dpi > WIDE_FIG_MIN_IN


# ---------------------------------------------------------------------------
# Dựng
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self, src, out, mode):
        self.src, self.out, self.mode = src, out, mode
        self.doc = Document(TPL)
        clear_body(self.doc)
        set_columns(self.doc.sections[0], 1)
        self.cols = 1
        self.after_head = True
        self.link = mode == "main"
        self.body_cols = 2 if mode == "main" else 1
        self.stats = {"tables": 0, "figures": 0, "equations": 0, "algorithms": 0}
        self.eq = 0

    # --- section
    def switch(self, n):
        if self.cols != n:
            s = self.doc.add_section(WD_SECTION.CONTINUOUS)
            set_columns(s, n)
            self.cols = n
            shrink_break(self.doc)

    def P(self, style, text=None, autonum_off=False, **kw):
        p = self.doc.add_paragraph(style=style)
        if autonum_off:
            no_autonum(p)
        if text is not None:
            emit(p, text, **kw)
        return p

    def width(self):
        return FULL_IN if self.cols == 1 else COL_IN

    # --- khối
    def heading1(self, t):
        self.switch(self.body_cols)
        m = re.match(r"((?:[IVX]+|S\d+)\.)\s*(.+)", t)
        # khoảng trắng thay cho tab: tab + thụt treo 280 twip làm "VII.DISCUSSION" dính
        # chữ vì "VII." rộng hơn 0,19 in ở Helvetica đậm 9 pt
        txt = f"{m.group(1)} {m.group(2).upper()}" if m else t.upper()
        p = self.P("H1_List (Space)", autonum_off=True)
        emit(p, txt)
        set_indent(p, left=0, first=0)
        self.after_head = True

    def heading2(self, t):
        self.switch(self.body_cols)
        p = self.P("H2_Cont", autonum_off=True)
        emit(p, t.upper() if re.match(r"[A-Z]\.\s", t) else t)
        self.after_head = True

    def paragraph(self, text):
        self.switch(self.body_cols)
        p = self.P("PARA" if self.after_head else "PARA_Indent")
        emit(p, text, link_cites=self.link)
        if re.search(r"\S{28,}", text):
            # mã băm 40-64 ký tự không ngắt được: canh đều sẽ kéo giãn cả dòng trước
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self.after_head = False

    def note(self, text, width_in):
        p = self.doc.add_paragraph(style="PARA")
        pf = p.paragraph_format
        pf.first_line_indent = 0
        pf.space_before = Pt(2)
        pf.space_after = Pt(6)
        pf.line_spacing = 1.0
        set_indent(p, left=0, first=0)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        emit(p, text, link_cites=self.link, size=TABLE_PT)
        return p

    def table_block(self, caption, panels, note):
        """caption: 'TABLE n. …'; panels: [(nhãn hoặc None, rows)]; note: 'Note. …'."""
        need = max(natural_width(r) for _, r in panels)
        ncol = max(max(len(x) for x in r) for _, r in panels)
        wide = self.mode == "supp" or need > COL_IN or ncol >= 6
        self.switch(1 if wide else self.body_cols)
        total = FULL_IN if wide else COL_IN
        cap = self.P("Table Caption", autonum_off=True)
        set_indent(cap, left=0, first=0)
        cap.alignment = WD_ALIGN_PARAGRAPH.LEFT if wide else WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.keep_with_next = True
        emit(cap, caption, link_cites=self.link)
        for label, rows in panels:
            if label:
                lp = self.doc.add_paragraph(style="PARA")
                set_indent(lp, left=0, first=0)
                lp.paragraph_format.space_before = Pt(3)
                lp.paragraph_format.space_after = Pt(1)
                lp.paragraph_format.keep_with_next = True
                _style_run(lp.add_run(label), size=TABLE_PT + 1)
            t = add_table(self.doc, rows, total, self.link)
            self.stats["tables"] += 1
        if note:
            # dòng Note đi cùng hàng cuối của bảng, kẻo bị đẩy một mình sang trang sau
            for cell in t.rows[-1].cells:
                for cp in cell.paragraphs:
                    cp.paragraph_format.keep_with_next = True
            self.note(note, total)
        else:
            sp = self.doc.add_paragraph(style="PARA")
            sp.paragraph_format.space_after = Pt(4)
        if wide:
            self.switch(self.body_cols)
        self.after_head = False

    def figure(self, path, caption):
        png = os.path.join(os.path.dirname(self.src), path)
        wide = fig_is_wide(png)
        self.switch(1 if wide else self.body_cols)
        w = FULL_IN if wide else COL_IN
        self.doc.add_picture(png, width=Inches(w))
        pic = self.doc.paragraphs[-1]
        pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.keep_with_next = True
        pic.paragraph_format.first_line_indent = 0
        cap = self.P("Fig Caption", autonum_off=True)
        set_indent(cap, left=0, first=0)
        # style Fig Caption của template bật keepNext: hình rộng + chú thích + bảng rộng
        # kế tiếp bị xích lại thành một khối quá cao, Word đẩy cả khối sang trang sau
        # và để trống nửa trang. Chú thích hình là phần cuối của khối hình.
        cap.paragraph_format.keep_with_next = False
        cap.paragraph_format.space_after = Pt(8)
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        emit(cap, caption, link_cites=self.link)
        self.stats["figures"] += 1
        if wide:
            self.switch(self.body_cols)
        self.after_head = False

    def equation(self, latex):
        """Phương trình trưng bày, số hiệu (n) sát lề phải.

        Dựng bằng bảng không viền hai cột: ô trái chứa m:oMathPara (toán trưng bày
        thật, phân số cỡ đầy đủ), ô phải chứa số hiệu canh phải. Đã thử hai cách khác
        và bỏ: (1) m:oMath chung đoạn với tab và số hiệu -> Word coi là toán trong dòng,
        phân số bị thu nhỏ; (2) mảng phương trình với '#(n)' -> Word for Mac in số hiệu
        sát ngay sau phương trình thay vì ở lề. Dòng quá rộng được ngắt ở mức ngoài
        cùng (mathrun.break_long), \\qquad tách dòng (split_display); chỉ dòng cuối
        mang số."""
        self.switch(self.body_cols)
        self.eq += 1
        num_in = 0.3
        eq_in = self.width() - num_in
        lines = []
        for part in split_display(latex):
            lines += break_long(part, eq_in / EM_IN)
        t = self.doc.add_table(rows=len(lines), cols=2)
        for row in t.rows:                    # ô bảng mặc định style Normal 12 pt: toán
            for cell in row.cells:            # sẽ to hơn chữ thân bài 10 pt
                cell.paragraphs[0].style = self.doc.styles["PARA"]
        tbl = t._tbl
        tblPr = tbl.tblPr
        for old in tblPr.findall(qn("w:tblStyle")):
            tblPr.remove(old)
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), str(int(self.width() * 1440)))
        w.set(qn("w:type"), "dxa")
        tblPr.append(w)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        mar = parse_xml(f'<w:tblCellMar xmlns:w="{W_NS}"><w:left w:w="0" w:type="dxa"/>'
                        f'<w:right w:w="0" w:type="dxa"/></w:tblCellMar>')
        tblPr.append(mar)
        cap = OxmlElement("w:tblCaption")
        cap.set(qn("w:val"), "equation")
        tblPr.append(cap)
        for gc, wi in zip(tbl.find(qn("w:tblGrid")).findall(qn("w:gridCol")), (eq_in, num_in)):
            gc.set(qn("w:w"), str(int(wi * 1440)))
        for r, part in enumerate(lines):
            last = r == len(lines) - 1
            row = t.rows[r]
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            cs.set(qn("w:val"), "true")
            trPr.append(cs)
            for j, (cell, wi) in enumerate(zip(row.cells, (eq_in, num_in))):
                cell.width = Inches(wi)
                tcPr = cell._tc.get_or_add_tcPr()
                va = OxmlElement("w:vAlign")
                va.set(qn("w:val"), "center")
                tcPr.append(va)
                p = cell.paragraphs[0]
                pf = p.paragraph_format
                pf.space_before = Pt(2 if r == 0 else 0)
                pf.space_after = Pt(2 if last else 0)
                pf.line_spacing = 1.0
                pf.first_line_indent = 0
                pf.keep_with_next = not last
                if j == 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p._p.append(parse_xml(display_omml(part)))
                elif last:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    p.add_run(f"({self.eq})")
        self.stats["equations"] += 1
        self.after_head = False

    # --- front matter
    def front_main(self, lines):
        title = lines[0][2:].strip()
        self.P("DOP", "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.")
        self.P("DOI", "Digital Object Identifier 10.1109/ACCESS.2024.Doi Number")
        self.P("Paper Title", title)
        i = 1
        while not lines[i].strip():
            i += 1
        self.P("AU", lines[i].strip())
        i += 1
        affs, ca = [], None
        while i < len(lines) and not lines[i].startswith("## "):
            s = lines[i].strip()
            if s.startswith("<sup>"):
                affs.append(s)
            elif s.startswith("Corresponding author"):
                ca = s
            i += 1
        for k, a in enumerate(affs):
            self.P("PI_No Space" if k < len(affs) - 1 else "PI", a)
        if ca:
            self.P("PI", ca)
        # ABSTRACT + INDEX TERMS
        assert lines[i].strip() == "## ABSTRACT", lines[i]
        i += 1
        abstract, it = None, None
        while i < len(lines) and not re.match(r"## [IVX]+\.", lines[i]):
            s = lines[i].strip()
            if s.startswith("INDEX TERMS"):
                it = s[len("INDEX TERMS"):].strip()
            elif s and abstract is None:
                abstract = s
            i += 1
        p = self.doc.add_paragraph(style="Abstract")
        p.add_run("ABSTRACT ").bold = True
        emit(p, abstract)
        p = self.doc.add_paragraph(style="IT")
        p.add_run("INDEX TERMS ").bold = True
        emit(p, it)
        self.switch(2)
        return i

    def front_supp(self, lines):
        p = self.P("Paper Title", lines[0][2:].strip())
        for r in p.runs:
            r.font.size = Pt(16)
        return 1

    # --- vòng chính
    def build(self):
        lines = open(self.src, encoding="utf-8").read().split("\n")
        i = self.front_main(lines) if self.mode == "main" else self.front_supp(lines)
        n = len(lines)
        in_refs = False
        while i < n:
            ln = lines[i]
            s = ln.strip()
            if not s or s == "---":
                i += 1
                continue
            if ln.startswith("## "):
                t = ln[3:].strip()
                in_refs = t.upper() == "REFERENCES"
                self.heading1(t)
                i += 1
                continue
            if ln.startswith("### "):
                self.heading2(ln[4:].strip())
                i += 1
                continue
            if s.startswith("$$"):
                self.equation(s.strip("$").strip())
                i += 1
                continue
            if s.startswith("```"):
                j = i + 1
                blk = []
                while j < n and not lines[j].strip().startswith("```"):
                    blk.append(lines[j])
                    j += 1
                self.switch(self.body_cols)
                add_algorithm(self.doc, blk, self.width())
                sp = self.doc.add_paragraph(style="PARA")
                sp.paragraph_format.space_after = Pt(2)
                self.stats["algorithms"] += 1
                self.after_head = False
                i = j + 1
                continue
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
            if m:
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                cap = lines[j].strip()
                assert cap.startswith("FIGURE"), f"hình {m.group(1)} thiếu chú thích: {cap[:60]}"
                self.figure(m.group(1), cap)
                i = j + 1
                continue
            if re.match(r"TABLE S?\d+\.", s):
                caption, panels, note = s, [], None
                j, label = i + 1, None
                while j < n:
                    t = lines[j].strip()
                    if not t:
                        j += 1
                        continue
                    if re.fullmatch(r"\([a-z]\)", t):
                        label = t
                        j += 1
                        continue
                    if t.startswith("|"):
                        blk = []
                        while j < n and lines[j].strip().startswith("|"):
                            blk.append(lines[j])
                            j += 1
                        panels.append((label, parse_table(blk)))
                        label = None
                        continue
                    if t.startswith("Note.") and panels:
                        note = t
                        j += 1
                    break
                assert panels, f"{caption[:40]}: không thấy bảng"
                self.table_block(caption, panels, note)
                i = j
                continue
            if s.startswith("|"):
                raise ValueError(f"bảng không có chú thích TABLE ở dòng {i + 1}")
            # đoạn văn (một dòng một đoạn trong các draft này)
            if in_refs:
                self.switch(self.body_cols)
                p = self.P("Ref")
                emit(p, s, math=False)
                mref = re.match(r"\[(\d+)\]", s)
                if mref:
                    add_bookmark(p, ref_anchor(mref.group(1)), 1000 + int(mref.group(1)))
            else:
                self.paragraph(s)
            i += 1
        if self.mode == "main":
            # ngắt section liên tục ở cuối để Word cân hai cột của trang tài liệu tham khảo
            self.switch(1)
        order_tblpr(self.doc)
        order_ppr(self.doc)
        self.doc.save(self.out)
        return self.stats


# ---------------------------------------------------------------------------
# Kiểm không cần Word
# ---------------------------------------------------------------------------
NS = {"w": W_NS, "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
EMU = 914400


def md_counts(src):
    txt = open(src, encoding="utf-8").read()
    lines = txt.split("\n")
    ntab = 0
    k = 0
    while k < len(lines):
        if lines[k].startswith("|"):
            ntab += 1
            while k < len(lines) and lines[k].startswith("|"):
                k += 1
        k += 1
    return {"tables": ntab, "figures": len(re.findall(r"^!\[", txt, re.M)),
            "equations": len(re.findall(r"^\$\$", txt, re.M)),
            "algorithms": txt.count("```") // 2,
            "cites": len(re.findall(r"\[\d+\]", txt.split("## REFERENCES")[0])),
            "refs": len(re.findall(r"^\[\d+\]", txt, re.M)),
            "tbd": len(re.findall(r"\[TBD", txt))}


def check(src, out, mode):
    from lxml import etree
    z = zipfile.ZipFile(out)
    xml = z.read("word/document.xml")
    root = etree.fromstring(xml)
    body = root.find("w:body", NS)
    want = md_counts(src)
    res, problems = {}, []

    tbls = body.findall(".//w:tbl", NS)
    kinds = {"data": 0, "equation": 0, "algorithm": 0}
    for t in tbls:
        cap = t.find("w:tblPr/w:tblCaption", NS)
        first = "".join(x.text or "" for x in t.findall(".//w:t", NS))[:12]
        if cap is not None and cap.get(f"{{{W_NS}}}val") == "equation":
            kinds["equation"] += 1
        elif first.startswith("Algorithm"):
            kinds["algorithm"] += 1
        else:
            kinds["data"] += 1
    res["table_kinds"] = kinds
    res["tables_docx"] = kinds["data"]
    if kinds["equation"] != want["equations"] or kinds["algorithm"] != want["algorithms"]:
        problems.append(f"phương trình/thuật toán: {kinds} so với md {want['equations']}/{want['algorithms']}")
    res["tables_md"] = want["tables"]
    if res["tables_docx"] != want["tables"]:
        problems.append(f"bảng: docx {res['tables_docx']} ≠ md {want['tables']}")

    # hình: số lượng, bề rộng so với section chứa nó
    secw = []                                  # bề rộng cột của section theo thứ tự
    paras = list(body.iter(f"{{{W_NS}}}p"))
    cur_pics, fig_rows = [], []
    for p in body.iter():
        if p.tag == f"{{{W_NS}}}drawing":
            ext = p.find(".//wp:extent", NS)
            cur_pics.append(int(ext.get("cx")))
        if p.tag == f"{{{W_NS}}}sectPr":
            cols = p.find("w:cols", NS)
            num = int(cols.get(f"{{{W_NS}}}num", "1")) if cols is not None else 1
            pgw = int(p.find("w:pgSz", NS).get(f"{{{W_NS}}}w"))
            mar = p.find("w:pgMar", NS)
            avail = (pgw - int(mar.get(f"{{{W_NS}}}left")) - int(mar.get(f"{{{W_NS}}}right"))) / 1440
            space = int(cols.get(f"{{{W_NS}}}space", "0")) / 1440 if cols is not None else 0
            colw = (avail - (num - 1) * space) / num
            for cx in cur_pics:
                fig_rows.append((cx / EMU, colw, num))
            secw.append((num, round(colw, 3)))
            cur_pics = []
    res["figures_docx"] = len(fig_rows)
    res["figures_md"] = want["figures"]
    if len(fig_rows) != want["figures"]:
        problems.append(f"hình: docx {len(fig_rows)} ≠ md {want['figures']}")
    res["figure_widths_in"] = [(round(w, 2), round(c, 2), n) for w, c, n in fig_rows]
    for w, c, num in fig_rows:
        if w > c + 0.005:
            problems.append(f"hình rộng {w:.2f} in > cột {c:.2f} in")

    # ảnh nhúng đúng file, đúng thứ tự (md5 ảnh trong word/media so với PNG gốc)
    import hashlib
    rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    blips = [b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
             for b in body.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")]
    srcdir = os.path.dirname(src)
    pngs = re.findall(r"^!\[[^\]]*\]\(([^)]+)\)", open(src, encoding="utf-8").read(), re.M)
    md5 = lambda b: hashlib.md5(b).hexdigest()
    got = [md5(z.read("word/" + rmap[b])) for b in blips]
    exp = [md5(open(os.path.join(srcdir, f), "rb").read()) for f in pngs]
    res["images_match_source_in_order"] = got == exp
    if got != exp:
        problems.append("ảnh nhúng không khớp PNG gốc hoặc sai thứ tự")

    # phương trình
    res["omath"] = len(body.findall(".//m:oMath", NS))

    # neo trích dẫn
    anchors = [h.get(f"{{{W_NS}}}anchor") for h in body.iter(f"{{{W_NS}}}hyperlink")]
    marks = {b.get(f"{{{W_NS}}}name") for b in body.iter(f"{{{W_NS}}}bookmarkStart")}
    dead = sorted({a for a in anchors if a not in marks})
    res["cite_links"], res["bookmarks"] = len(anchors), len([m for m in marks if m.startswith("ref")])
    res["cites_md"] = want["cites"] if mode == "main" else 0
    if mode == "main" and len(anchors) != want["cites"]:
        problems.append(f"liên kết trích dẫn {len(anchors)} ≠ [n] trong md {want['cites']}")
    if dead:
        problems.append(f"neo chết: {dead[:10]}")
    if mode == "main" and res["bookmarks"] != want["refs"]:
        problems.append(f"bookmark {res['bookmarks']} ≠ mục tài liệu {want['refs']}")

    # chữ còn sót
    text = "".join(t.text or "" for t in body.iter(f"{{{W_NS}}}t"))
    mtext = "".join(t.text or "" for t in body.iter(f"{{{NS['m']}}}t"))
    leftovers = {}
    for pat in [r"\\[a-zA-Z]+", r"\$", r"\*\*", r"<sup>", r"\^\{", r"_\{", r"!\[",
                r"(?<![\w/.])[A-Za-zμσŷ]̂?_[A-Za-z0-9]\b"]:
        hits = re.findall(pat, text + mtext)
        if hits:
            leftovers[pat] = hits[:5]
    res["leftovers"] = leftovers
    if leftovers:
        problems.append(f"chữ LaTeX/markdown còn sót: {leftovers}")
    res["underscore_tokens"] = sorted(set(re.findall(r"\b\w*_\w+\b", text)))

    # tô vàng TBD
    hl = 0
    for r in body.iter(f"{{{W_NS}}}r"):
        h = r.find("w:rPr/w:highlight", NS)
        t = "".join(x.text or "" for x in r.findall("w:t", NS))
        if h is not None and t.startswith("[TBD"):
            hl += 1
    res["tbd_highlighted"], res["tbd_md"] = hl, want["tbd"]
    if hl != want["tbd"]:
        problems.append(f"TBD tô vàng {hl} ≠ md {want['tbd']}")

    # tiêu đề / chú thích tự đánh số mà không tắt
    auto = []
    for p in paras:
        st = p.find("w:pPr/w:pStyle", NS)
        if st is None:
            continue
        name = st.get(f"{{{W_NS}}}val")
        if re.match(r"(H1|H2|H3|FigCaption|Fig)", name or ""):
            nid = p.find("w:pPr/w:numPr/w:numId", NS)
            if nid is None or nid.get(f"{{{W_NS}}}val") != "0":
                auto.append(name)
            else:                       # con của pPr phải theo thứ tự schema
                pPr = p.find("w:pPr", NS)
                kids = [c.tag.split('}')[1] for c in pPr]
                ranks = [_PPR_ORDER.index(k) for k in kids if k in _PPR_ORDER]
                if ranks != sorted(ranks):
                    auto.append(name + "(thứ tự)")
    bad_order = 0
    for pPr in body.iter(f"{{{W_NS}}}pPr"):
        kids = [c.tag.split('}')[1] for c in pPr]
        ranks = [_PPR_ORDER.index(k) for k in kids if k in _PPR_ORDER]
        bad_order += ranks != sorted(ranks)
    res["ppr_out_of_order"] = bad_order
    if bad_order:
        problems.append(f"{bad_order} w:pPr sai thứ tự schema")
    res["autonum_left"] = auto
    if auto:
        problems.append(f"đoạn tự đánh số chưa tắt: {auto[:5]}")

    # thứ tự chú thích: bảng có caption ngay trước, hình có caption ngay sau
    seq = []
    for el in body:
        tag = el.tag.split("}")[1]
        if tag == "tbl":
            seq.append("TBL")
        elif tag == "p":
            st = el.find("w:pPr/w:pStyle", NS)
            nm = st.get(f"{{{W_NS}}}val") if st is not None else ""
            if el.find(".//w:drawing", NS) is not None:
                seq.append("PIC")
            elif nm in ("TableCaption", "Table Caption"):
                seq.append("TCAP")
            elif nm in ("FigCaption", "Fig Caption"):
                seq.append("FCAP")
            else:
                seq.append("P")
    bad = 0
    for k, x in enumerate(seq):
        if x == "PIC" and (k + 1 >= len(seq) or seq[k + 1] != "FCAP"):
            bad += 1
    res["pic_without_caption_below"] = bad
    if bad:
        problems.append(f"{bad} hình không có chú thích ngay dưới")
    res["sections"] = secw
    res["problems"] = problems
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="chỉ kiểm, không dựng lại")
    ap.add_argument("--only", choices=list(JOBS))
    a = ap.parse_args(argv)
    ok = True
    for mode, (src, out) in JOBS.items():
        if a.only and mode != a.only:
            continue
        if not a.check:
            st = Builder(src, out, mode).build()
            print(f"-> {os.path.relpath(out, REPO)}: {st}")
        res = check(src, out, mode)
        for k, v in res.items():
            if k != "underscore_tokens":
                print(f"   {k}: {v}")
        print(f"   underscore_tokens (giữ nguyên, kiểm bằng mắt): {res['underscore_tokens']}")
        ok &= not res["problems"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
