"""Render toán LaTeX sang Word.

Hai đường:
  * `emit_inline(par, latex)` — toán trong dòng, dựng bằng run có vertAlign
    subscript/superscript thật (KHÔNG dùng "_"/"^" ASCII).
  * `add_display_equation(doc, ...)` — phương trình trưng bày, dựng bằng OMML
    (object phương trình thật của Word) kèm số hiệu (1) canh phải theo lệ IEEE.

Bug đã sửa: bản cũ (`build_docx._clean_math`) xoá dấu ngoặc nhọn TRƯỚC rồi mới
xoá lệnh LaTeX, nên `\\text{MSE}` biến thành `\\textMSE` rồi bị regex
`\\\\[a-zA-Z]+` nuốt trọn — mất luôn nội dung. Tương tự `\\frac`, `\\min`,
`\\max`, `\\log`, `\\le`. Ở đây lệnh được tra bằng từ điển TRƯỚC, và nội dung
trong ngoặc luôn được giữ.
"""
import re

from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# lệnh LaTeX -> ký tự Unicode. Lệnh KHÔNG có trong bảng được giữ nguyên tên
# (bỏ dấu \), nên \min \max \log tự động ra "min" "max" "log".
SYMS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "sigma": "σ", "mu": "μ",
    "times": "×", "cdot": "·", "propto": "∝", "Delta": "Δ", "delta": "δ",
    "sqrt": "√", "sum": "Σ", "in": "∈", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "approx": "≈", "pm": "±", "to": "→",
    "ldots": "…", "dots": "…", "infty": "∞", "neq": "≠",
    "qquad": "\u2003\u2003", "quad": "\u2003", ",": "\u2009", ";": "\u2009",
    " ": " ", "!": "",
}


def _expand(s):
    """Đổi lệnh LaTeX thành Unicode, GIỮ nội dung trong \\text{...}."""
    # \text{MSE} -> MSE  (giữ nội dung — đây là chỗ bản cũ làm mất chữ)
    s = re.sub(r"\\(?:text|mathrm|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    # \frac{a}{b} -> a/b  (chỉ dùng cho toán trong dòng; display đi đường OMML)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\([,;! ])", lambda m: SYMS.get(m.group(1), ""), s)
    s = re.sub(r"\\([a-zA-Z]+)", lambda m: SYMS.get(m.group(1), m.group(1)), s)
    s = s.replace(r"\{", "{").replace(r"\}", "}").replace("\\", "")
    return s


def _tokens(s):
    """Cắt chuỗi đã expand thành [(text, level)], level ∈ {'', 'sub', 'sup'}."""
    out, i, n = [], 0, len(s)
    buf = ""
    while i < n:
        c = s[i]
        if c in "_^":
            if buf:
                out.append((buf, ""))
                buf = ""
            lvl = "sub" if c == "_" else "sup"
            i += 1
            if i < n and s[i] == "{":
                depth, j = 1, i + 1
                while j < n and depth:
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                    j += 1
                out.append((s[i + 1:j - 1], lvl))
                i = j
            elif i < n:
                out.append((s[i], lvl))
                i += 1
        else:
            buf += c
            i += 1
    if buf:
        out.append((buf, ""))
    return out


def emit_inline(par, latex):
    """Đổ toán trong dòng vào `par`; trả về danh sách run vừa tạo."""
    out = []
    for text, lvl in _tokens(_expand(latex)):
        if not text:
            continue
        r = par.add_run(text)
        if lvl == "sub":
            r.font.subscript = True
        elif lvl == "sup":
            r.font.superscript = True
        out.append(r)
    return out


# --------------------------------------------------------------- OMML
def _r(t, sty="i"):
    """Một run toán OMML. sty='i' nghiêng (biến), 'p' đứng (tên hàm/số)."""
    t = (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    pr = f'<m:rPr><m:sty m:val="{sty}"/></m:rPr>' if sty else ""
    return f"<m:r>{pr}<m:t xml:space=\"preserve\">{t}</m:t></m:r>"


def _sub(base, sub):
    return (f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>")


def _sup(base, sup):
    return (f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>")


def _subsup(base, sub, sup):
    return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
            f"<m:sup>{sup}</m:sup></m:sSubSup>")


def _nary(chr_, sub, sup, body):
    return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/>'
            f'<m:limLoc m:val="undOvr"/><m:supHide m:val="0"/>'
            f'<m:subHide m:val="0"/></m:naryPr>'
            f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>")


def _rad(body):
    return (f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr>'
            f"<m:deg/><m:e>{body}</m:e></m:rad>")


def _frac(num, den):
    return f"<m:f><m:fPr/><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def fitness_equation_omml():
    """OMML của công thức tail-weighted fitness (phương trình (1) của bài).

        F_tail = √( Σ_{b=1}^{B} w_b · MSE_b ),   w_b = n_b^{1-α} / Σ_j n_j^{1-α}
    """
    # F_tail =
    lhs = _sub(_r("F"), _r("tail", "p"))
    # Σ_{b=1}^{B} w_b · MSE_b
    sum_body = (_sub(_r("w"), _r("b")) + _r("·", "p")
                + _sub(_r("MSE", "p"), _r("b")))
    sum1 = _nary("∑", _r("b", "i") + _r("=1", "p"), _r("B"), sum_body)
    left = lhs + _r(" = ", "p") + _rad(sum1)

    # w_b = n_b^{1-α} / Σ_{j=1}^{B} n_j^{1-α}
    num = _subsup(_r("n"), _r("b"), _r("1", "p") + _r("−", "p") + _r("α"))
    den = _nary("∑", _r("j", "i") + _r("=1", "p"), _r("B"),
                _subsup(_r("n"), _r("j"), _r("1", "p") + _r("−", "p") + _r("α")))
    right = _sub(_r("w"), _r("b")) + _r(" = ", "p") + _frac(num, den)

    return (f'<m:oMath xmlns:m="{M_NS}">'
            f'{left}{_r(",", "p")}{_r("\u2003\u2003", "p")}{right}'
            f'</m:oMath>')


def cost_equation_omml():
    """OMML của công thức chi phí quyết định (phương trình (2) của bài).

        Cost(α, K) = Σ_r n_r c_r MAE_r(α) / Σ_r n_r c_r
    """
    lhs = (_r("Cost", "p") + _r("(", "p") + _r("α") + _r(", ", "p")
           + _r("K") + _r(")", "p"))
    term = (_sub(_r("n"), _r("r")) + _sub(_r("c"), _r("r"))
            + _sub(_r("MAE", "p"), _r("r")) + _r("(", "p") + _r("α") + _r(")", "p"))
    num = _nary("∑", _r("r"), "", term)
    den = _nary("∑", _r("r"), "", _sub(_r("n"), _r("r")) + _sub(_r("c"), _r("r")))
    return (f'<m:oMath xmlns:m="{M_NS}">'
            f'{lhs}{_r(" = ", "p")}{_frac(num, den)}'
            f'</m:oMath>')


# Bảng tra: đoạn LaTeX trong draft -> hàm dựng OMML. Trước đây build_docx.py
# gọi thẳng fitness_equation_omml() cho MỌI khối $$, nên thêm phương trình thứ
# hai vào bài sẽ render ra bản sao của phương trình (1) mà không báo lỗi gì.
# Thêm phương trình mới thì thêm một khoá nhận dạng ở đây.
_EQUATIONS = [("F_{\\text{tail}}", fitness_equation_omml),
              ("\\text{Cost}", cost_equation_omml)]


def display_equation_for(latex):
    """Chọn bộ dựng OMML theo nội dung LaTeX. Không nhận ra thì BÁO LỖI —
    thà dừng build còn hơn lặng lẽ in ra một phương trình khác."""
    for key, fn in _EQUATIONS:
        if key in latex:
            return fn()
    raise ValueError(
        "Không nhận ra phương trình trưng bày:\n  " + latex.strip()[:120] +
        "\nThêm bộ dựng OMML vào _EQUATIONS trong src/mathrun.py.")


def add_display_equation(doc, omml, number=None, style="PARA"):
    """Chèn phương trình trưng bày: canh giữa, số hiệu (n) canh phải bằng tab.

    IEEE Access đánh số mọi phương trình trưng bày; bài có tham chiếu tới nó."""
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.first_line_indent = 0
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(omml))
    if number is not None:
        p.add_run("\t")
        p.add_run(f"({number})")
        # tab phải ở mép cột để số hiệu nằm sát lề phải
        tabs = parse_xml(
            f'<w:tabs xmlns:w="{W_NS}">'
            f'<w:tab w:val="right" w:pos="4680"/></w:tabs>')
        p._p.get_or_add_pPr().append(tabs)
    return p
