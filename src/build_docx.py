# -*- coding: utf-8 -*-
"""Dựng bản .docx đúng format IEEE Access từ paper/draft_numbered.md.

Dùng CHÍNH template làm gốc (giữ khổ giấy, lề, header có logo IEEE Access, và
toàn bộ style) rồi đổ nội dung vào. Luật format nằm ở src/docx_ieee.py.

Bố cục theo template:
    section 0 (1 CỘT): DOP, DOI, tiêu đề, tác giả, affiliation, corresponding,
                       footnote tài trợ, ABSTRACT, INDEX TERMS
    section 1 (2 CỘT): thân bài
    ... xen kẽ section 1 cột cho hình/bảng khổ đôi rồi trả lại 2 cột.

Chạy: PYTHONPATH=src python3 src/build_docx.py
Xuất: paper/GA4XGB_IEEE_Access.docx
"""
import os
import re

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION

from docx_ieee import (clear_body, set_columns, no_autonum, set_indent,
                       format_table_ieee, repeat_header, shrink_break,
                       add_bookmark, add_cite_link, ref_anchor, FULL_W, COL_W)
from mathrun import emit_inline, add_display_equation, fitness_equation_omml


def strip_comments(text):
    """Gỡ chú thích <!-- ... --> nằm GIỮA dòng.

    Vòng lặp chính chỉ bỏ qua comment khi nó đứng ĐẦU dòng, nên chú thích viết
    lẫn trong đoạn văn từng lọt nguyên văn vào bản .docx nộp đi."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()

SRC = "paper/draft_numbered.md"
TPL = "paper/Access-Template-2024.docx"
OUT = "paper/GA4XGB_IEEE_Access.docx"
FIGDIR = "paper/figures"

# Hình khổ đôi (7.16in) vs khổ một cột (3.5in) — theo hướng dẫn IEEE Access.
WIDE_FIGS = {"FIGURE 5", "FIGURE 6"}
# Bảng khổ đôi: Table 2 (mô tả thuộc tính dài), Table 5 (6 cột nhãn dài).
WIDE_TABLES = {2, 5}

FIGFILE = {
    "FIGURE 1": "fig1_flowchart.png",
    "FIGURE 2": "fig2_crossover.png",
    "FIGURE 3": "fig3_weight_profiles.png",
    "FIGURE 4": "fig4_score_histogram.png",
    "FIGURE 5": "fig5_convergence.png",
    "FIGURE 6": "fig6_mae_by_bin.png",
    "FIGURE 7": "fig7_feature_importance.png",
}
FIGCAP = {
    "FIGURE 1": "FIGURE 1. Key steps of the proposed GA4XGB algorithm for optimizing "
                "XGBoost hyperparameters in a student score prediction system.",
    "FIGURE 2": "FIGURE 2. Illustration of the real-coded evolution process in GA4XGB. "
                "BLX-α samples each offspring gene from an interval extended by $\\alpha_x$ "
                "beyond the parents, and Gaussian mutation perturbs genes in place; "
                "neither operator quantizes the search space.",
    "FIGURE 3": "FIGURE 3. Profiles of the tail-weighted fitness weights $w_b$ for "
                "α ∈ {0, 0.5, 1}, against the score distribution. α = 0 reproduces the "
                "conventional RMSE; α = 1 weights all bins equally.",
    "FIGURE 4": "FIGURE 4. Distribution of raw HSA scores in the dataset (n = 57,174). "
                "Bin edges used by the fitness function are marked; the shaded areas are "
                "the low-score (below 60) and high-score (100 or above) regions on which "
                "tail behavior is reported throughout this paper.",
    "FIGURE 5": "FIGURE 5. Convergence of GA4XGB variants. (a) Best fitness per "
                "generation, normalized within each run. (b) Cross-validated RMSE of the "
                "incumbent against the default-configuration reference.",
    "FIGURE 6": "FIGURE 6. (a) MAE across score bins: conventional methods share a 3.4× "
                "U-shape, whereas GA4XGB with loss weighting is near-flat. (b) Relative "
                "change in MAE against GA-RMSE with 95% bootstrap confidence intervals. "
                "(c) The (α, β) frontier between aggregate and tail accuracy.",
    "FIGURE 7": "FIGURE 7. Top-20 gain-based feature importances of the selected model, "
                "coloured by feature group. Mathematics grades and province indicators "
                "dominate.",
}

FRONT_AUTHORS = "NGUYEN-BA-TIEN1,2, NGO-THI-THU-TRANG3, AND HA-NAM NGUYEN4"
FRONT_AFF = [
    "1International School, Vietnam National University, Hanoi 122300, Vietnam",
    "2Institute of Digital Education and Testing, Vietnam National University, "
    "Hanoi 122300, Vietnam",
    "3Posts and Telecommunications Institute of Technology, Hanoi 122300, Vietnam",
    "4Information Technology Department, Electric Power University, Hanoi 122300, Vietnam",
]
FRONT_CA = "Corresponding author: Ha-Nam Nguyen (e-mail: namnhvn@epu.edu.vn)."
FRONT_FN = ("This work received no specific grant from any funding agency in the "
            "public, commercial, or not-for-profit sectors.")


def _emit_text(par, c, link_cites):
    """Đổ chữ thường; nếu link_cites thì [n] thành liên kết nội bộ tới ref n."""
    if not link_cites or "[" not in c:
        return [par.add_run(c)]
    out = []
    for piece in re.split(r"(\[\d+\])", c):
        if not piece:
            continue
        m = re.fullmatch(r"\[(\d+)\]", piece)
        if m:
            add_cite_link(par, piece, ref_anchor(m.group(1)))
        else:
            out.append(par.add_run(piece))
    return out


def _emit_seg(par, text, link_cites, bold=False, italic=False):
    """Một đoạn không còn markup đậm/nghiêng: tách $toán$ khỏi chữ thường."""
    for piece in re.split(r"(\$[^$]+\$)", text):
        if not piece:
            continue
        if len(piece) > 2 and piece[0] == "$" and piece[-1] == "$":
            runs = emit_inline(par, piece[1:-1])
        else:
            runs = _emit_text(par, piece, link_cites)
        for r in runs:
            if bold:
                r.bold = True
            if italic:
                r.italic = True


def add_md_runs(par, text, link_cites=False):
    """Đổ text có **đậm**, *nghiêng*, <i>nghiêng</i>, $toán$ thành các run.

    link_cites=True: mọi [n] trong đoạn trở thành liên kết bấm được xuống mục
    tài liệu tham khảo tương ứng (dùng cho thân bài, KHÔNG dùng cho mục REFERENCES).

    Toán trong $...$ đi qua mathrun.emit_inline -> subscript/superscript THẬT."""
    text = strip_comments(text)
    text = text.replace("<i>", "*").replace("</i>", "*")
    for chunk in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            _emit_seg(par, chunk[2:-2], link_cites, bold=True)
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            _emit_seg(par, chunk[1:-1], link_cites, italic=True)
        else:
            _emit_seg(par, chunk, link_cites)
    return par


def style_or_(doc, name, fallback="Normal"):
    try:
        doc.styles[name]
        return name
    except KeyError:
        return fallback


def col_widths(ncols, total_in, header):
    """Cột đầu (nhãn) rộng gấp đôi các cột số."""
    if ncols <= 2:
        return [total_in * 0.32, total_in * 0.68]
    first = total_in * (0.30 if ncols >= 5 else 0.34)
    rest = (total_in - first) / (ncols - 1)
    return [first] + [rest] * (ncols - 1)


def main():
    doc = Document(TPL)
    clear_body(doc)

    # ---- section 0: 1 cột cho toàn bộ front matter ----
    set_columns(doc.sections[0], 1)

    def P(style, text=None, autonum_off=False):
        p = doc.add_paragraph(style=style)
        if autonum_off:
            no_autonum(p)
        if text is not None:
            add_md_runs(p, text)
        return p

    lines = open(SRC).read().split("\n")
    title = lines[0][2:].strip()

    P("DOP", "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.")
    P("DOI", "Digital Object Identifier 10.1109/ACCESS.2024.Doi Number")
    P("Paper Title", title)
    # dòng tác giả: số affiliation dạng superscript (như template gốc)
    pau = doc.add_paragraph(style="AU")
    for piece in re.split(r"(\d+(?:,\d+)*)", FRONT_AUTHORS):
        if not piece:
            continue
        r = pau.add_run(piece)
        if piece[0].isdigit():
            r.font.superscript = True
    for k, a in enumerate(FRONT_AFF):
        pa = doc.add_paragraph(style="PI_No Space" if k < len(FRONT_AFF) - 1 else "PI")
        m = re.match(r"(\d+)(.*)", a)
        r = pa.add_run(m.group(1))
        r.font.superscript = True
        pa.add_run(m.group(2))
    P("PI", FRONT_CA)
    P("footnote text", FRONT_FN)

    # ABSTRACT + INDEX TERMS lấy từ draft
    abstract = next(l for l in lines if l.startswith("Accurate early prediction"))
    it = next(l for l in lines if l.startswith("**INDEX TERMS**"))
    p = doc.add_paragraph(style="Abstract")
    p.add_run("ABSTRACT ").bold = True
    add_md_runs(p, abstract)
    p = doc.add_paragraph(style="IT")
    p.add_run("INDEX TERMS ").bold = True
    add_md_runs(p, it.replace("**INDEX TERMS**", "").strip())

    # ---- chuyển sang 2 cột cho thân bài ----
    body = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(body, 2)
    shrink_break(doc)

    state = {"cols": 2, "tbl": 0, "eq": 0, "after_head": False}

    def switch(n):
        if state["cols"] != n:
            s = doc.add_section(WD_SECTION.CONTINUOUS)
            set_columns(s, n)
            state["cols"] = n
            shrink_break(doc)

    i, n = 0, len(lines)
    in_refs = False
    in_bios = False
    skip_front = True

    while i < n:
        ln = lines[i]

        if ln.startswith("<!--"):
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if ln.strip() in ("---", "") or ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("**Authors:**"):
            i += 1
            continue

        # ---------- H1 ----------
        if ln.startswith("## "):
            t = ln[3:].strip()
            if t.upper().startswith("ABSTRACT"):
                skip_front = True
                i += 1
                continue
            skip_front = False
            in_refs = t.upper().startswith("REFERENCES")
            in_bios = t.upper().startswith("AUTHOR BIOG")
            switch(2)
            if in_bios:
                # IEEE Access KHÔNG in tiêu đề trước phần tiểu sử — tiểu sử đi
                # ngay sau REFERENCES kèm ảnh. Đánh dấu trạng thái rồi bỏ qua.
                state["after_head"] = False
                i += 1
                continue
            m = re.match(r"([IVX]+)\.\s*(.+)", t)
            txt = f"{m.group(1)}.\t{m.group(2).upper()}" if m else t.upper()
            p = P("H1_List (Space)", autonum_off=True)
            add_md_runs(p, txt)
            set_indent(p, left=280, hanging=280)
            state["after_head"] = True
            i += 1
            continue

        if skip_front:
            i += 1
            continue

        # ---------- H2 / H3 ----------
        if ln.startswith("### "):
            t = ln[4:].strip()
            switch(2)
            if re.match(r"[A-Z]\.\s", t):                    # "A. TITLE" -> H2
                p = P("H2_Cont", autonum_off=True)
                add_md_runs(p, t.upper())
            else:                                            # tiêu đề mức 3
                p = P("H3", autonum_off=True)
                add_md_runs(p, t)
                for r in p.runs:      # style H3 bật <w:caps/>; câu dài viết hoa
                    r.font.all_caps = False   # toàn bộ thì rất khó đọc
            state["after_head"] = True
            i += 1
            continue

        # ---------- bảng ----------
        if ln.startswith("|"):
            blk = []
            while i < n and lines[i].startswith("|"):
                blk.append(lines[i])
                i += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in blk]
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            state["tbl"] += 1
            idx = state["tbl"]
            wide = idx in WIDE_TABLES
            switch(1 if wide else 2)

            # caption ĐỨNG TRÊN bảng (luật IEEE); trong draft nó nằm dưới -> tìm tới.
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n and lines[j].startswith("TABLE"):
                cap = doc.add_paragraph(style="Table Caption")
                add_md_runs(cap, lines[j])
                cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
                i = j + 1

            ncol = len(rows[0])
            total = 6.95 if wide else 3.33
            ws = col_widths(ncol, total, rows[0])
            t = doc.add_table(rows=len(rows), cols=ncol)
            for r_i, row in enumerate(rows):
                for c_j in range(ncol):
                    cell = t.cell(r_i, c_j)
                    cell.text = ""
                    p = cell.paragraphs[0]
                    p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if c_j == 0
                                   else WD_ALIGN_PARAGRAPH.CENTER)
                    add_md_runs(p, row[c_j] if c_j < len(row) else "")
                    if r_i == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for r in p.runs:
                            r.bold = True
            format_table_ieee(t, ws)
            repeat_header(t)
            if wide:
                switch(2)
            state["after_head"] = False
            continue

        # ---------- hình ----------
        m = re.match(r"\*\*\[(FIGURE \d)", ln)
        if m:
            key = m.group(1)
            wide = key in WIDE_FIGS
            switch(1 if wide else 2)
            png = os.path.join(FIGDIR, FIGFILE[key])
            if os.path.exists(png):
                doc.add_picture(png, width=Inches(6.95) if wide else Inches(3.33))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                ph = doc.add_paragraph(f"[thiếu {png}]")
                ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(style="Fig Caption")
            no_autonum(cap)          # style tự sinh "FIGURE n." -> tắt, ta gõ tay
            set_indent(cap, left=0, first=0)
            add_md_runs(cap, FIGCAP[key])
            if wide:
                switch(2)
            state["after_head"] = False
            i += 1
            continue

        # ---------- phương trình trưng bày ----------
        if ln.strip().startswith("$$"):
            switch(2)
            state["eq"] += 1
            add_display_equation(doc, fitness_equation_omml(), number=state["eq"])
            state["after_head"] = False
            i += 1
            continue

        # ---------- đoạn văn ----------
        blk = [ln]
        i += 1
        while i < n and lines[i].strip() and lines[i][0] not in "#|" and \
                not lines[i].startswith("**[") and not lines[i].startswith("TABLE"):
            blk.append(lines[i])
            i += 1
        text = " ".join(x.strip() for x in blk)

        switch(2)
        if in_bios:
            p = doc.add_paragraph(style=style_or_(doc, "AU_Bios"))
            add_md_runs(p, text)
        elif in_refs:
            p = doc.add_paragraph(style="Ref")
            add_md_runs(p, text)
            mref = re.match(r"\[(\d+)\]", text)
            if mref:
                add_bookmark(p, ref_anchor(mref.group(1)), 1000 + int(mref.group(1)))
        elif text.startswith("- "):
            p = doc.add_paragraph(style="PARA")
            add_md_runs(p, text[2:], link_cites=True)
        else:
            # PARA cho đoạn đầu sau tiêu đề, PARA_Indent cho các đoạn tiếp theo
            p = doc.add_paragraph(style="PARA" if state["after_head"] else "PARA_Indent")
            add_md_runs(p, text, link_cites=True)
            state["after_head"] = False

    doc.save(OUT)

    d2 = Document(OUT)
    words = sum(len(p.text.split()) for p in d2.paragraphs)
    print(f"-> {OUT}")
    print(f"   {len(d2.paragraphs)} đoạn | {len(d2.tables)} bảng | "
          f"{len(d2.inline_shapes)} hình | ~{words:,} từ")
    print("   sections:")
    from docx.oxml.ns import qn as _q
    for k, s in enumerate(d2.sections):
        c = s._sectPr.find(_q("w:cols"))
        print(f"     sec{k}: {c.get(_q('w:num')) or 1} cột")


if __name__ == "__main__":
    main()
