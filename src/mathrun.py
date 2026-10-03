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


# --------------------------------------------------------------- LaTeX -> OMML
# Bộ dịch chung cho phương trình trưng bày của bài cost-aware (draft_cost_v3.md).
# Vì sao không thêm từng hàm dựng tay như fitness_equation_omml: bài mới có sáu
# phương trình, dựng tay từng cái thì mỗi lần sửa chữ trong draft lại phải sửa mã,
# và lệch nhau mà không ai biết (đúng loại lỗi _EQUATIONS đã gặp). Ở đây dịch thẳng
# tập con LaTeX mà draft dùng; gặp lệnh lạ thì BÁO LỖI thay vì in tên lệnh ra trang.
# Không đụng tới các hàm cũ: build_docx.py vẫn chạy y như trước.

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_OPS = {
    "in": "∈", "notin": "∉", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "neq": "≠", "approx": "≈", "times": "×", "cdot": "·", "pm": "±", "mid": "∣",
    "to": "→", "infty": "∞", "propto": "∝", "subset": "⊂", "ldots": "…",
    "dots": "…", "{": "{", "}": "}",
}
_SPACES = {",": " ", ";": " ", ":": " ", "!": "", " ": " ",
           "quad": " ", "qquad": "  "}
_FUNCS = {"liminf": "lim inf", "limsup": "lim sup", "min": "min", "max": "max",
          "log": "log", "exp": "exp", "sin": "sin", "cos": "cos", "lim": "lim"}
_ACCENTS = {"hat": "̂", "bar": "̅", "tilde": "̃", "dot": "̇"}
_SIZERS = {"big", "Big", "bigg", "Bigg", "left", "right", "bigl", "bigr", "Bigl", "Bigr"}


def _lex(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 < n and s[i + 1].isalpha():
                j = i + 1
                while j < n and s[j].isalpha():
                    j += 1
                toks.append(("cmd", s[i + 1:j]))
                i = j
            else:
                toks.append(("cmd", s[i + 1:i + 2]))
                i += 2
        elif c in "{}^_":
            toks.append((c, c))
            i += 1
        elif c.isspace():
            i += 1
        else:
            toks.append(("ch", c))
            i += 1
    return toks


class _P:
    def __init__(self, toks):
        self.t, self.k = toks, 0

    def peek(self):
        return self.t[self.k] if self.k < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.k += 1
        return tok

    def group(self):
        """Đọc một đối số: {…} hoặc một token đơn."""
        typ, val = self.peek()
        if typ == "{":
            self.take()
            out = self.seq(stop="}")
            self.take()
            return out
        return self.atom()

    def seq(self, stop=None):
        out = []
        while True:
            typ, val = self.peek()
            if typ is None or (stop and typ == stop):
                return "".join(out)
            out.append(self.scripted())

    def scripted(self):
        base = self.atom()
        sub = sup = None
        while self.peek()[0] in ("_", "^"):
            typ, _ = self.take()
            arg = self.group()
            if typ == "_":
                sub = arg
            else:
                sup = arg
        if sub is not None and sup is not None:
            return _subsup(base, sub, sup)
        if sub is not None:
            return _sub(base, sub)
        if sup is not None:
            return _sup(base, sup)
        return base

    def atom(self):
        typ, val = self.take()
        if typ == "{":
            out = self.seq(stop="}")
            self.take()
            return out
        if typ == "ch":
            if val.isalpha():
                return _r(val)
            return _r({"-": "−", "*": "∗"}.get(val, val), "p")
        if typ == "cmd":
            if val in _SPACES:
                return _r(_SPACES[val], "p") if _SPACES[val] else ""
            if val in _SIZERS:
                return ""                    # \big( -> chỉ giữ dấu ngoặc
            if val in _GREEK:
                return _r(_GREEK[val])
            if val in _OPS:
                return _r(_OPS[val], "p")
            if val in _FUNCS:
                return _r(_FUNCS[val], "p")
            if val == "frac":
                return _frac(self.group(), self.group())
            if val == "sqrt":
                return _rad(self.group())
            if val in ("sum", "prod", "int"):
                return _r({"sum": "∑", "prod": "∏", "int": "∫"}[val], "p")
            if val in _ACCENTS:
                body = self.group()
                return (f'<m:acc><m:accPr><m:chr m:val="{_ACCENTS[val]}"/></m:accPr>'
                        f"<m:e>{body}</m:e></m:acc>")
            if val in ("mathrm", "text", "operatorname", "textrm"):
                return _r(_flat(self), "p")
            if val == "mathbb":
                txt = _flat(self)
                return _r({"1": "𝟙", "R": "ℝ", "E": "𝔼", "N": "ℕ"}.get(txt, txt), "p")
            raise ValueError(f"Lệnh LaTeX chưa hỗ trợ trong latex_to_omml: \\{val}")
        raise ValueError(f"Token lạ trong phương trình: {typ!r} {val!r}")


def _flat(p):
    """Đối số {…} dạng chữ thô (cho \\mathrm, \\text, \\mathbb)."""
    typ, val = p.take()
    if typ != "{":
        return val
    out = []
    while p.peek()[0] not in ("}", None):
        out.append(p.take()[1])
    p.take()
    return "".join(out)


def latex_to_omml(latex):
    """Chuỗi LaTeX (không có $$) -> một <m:oMath> hoàn chỉnh."""
    body = _P(_lex(latex.strip())).seq()
    return f'<m:oMath xmlns:m="{M_NS}">{body}</m:oMath>'


def split_display(latex):
    """Tách phương trình có \\qquad ở mức ngoài cùng thành nhiều dòng trưng bày.

    Vì sao: cột IEEE chỉ rộng 3,33 in; phương trình (1) ghép định nghĩa w_K và cost_K
    bằng \\qquad, đặt trên một dòng sẽ tràn cột. Tách đúng chỗ tác giả đã ngăn cách."""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(latex):
        if latex.startswith("\\qquad", i) and depth == 0:
            parts.append(cur)
            cur = ""
            i += len("\\qquad")
            continue
        c = latex[i]
        depth += (c == "{") - (c == "}")
        cur += c
        i += 1
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def display_omml(latex, number=None, size_pt=10):
    """Phương trình trưng bày THẬT (m:oMathPara), tuỳ chọn số hiệu (n) sát lề phải.

    Vì sao không dùng đoạn văn + tab như add_display_equation: một m:oMath nằm chung
    đoạn với chữ (tab, số hiệu) bị Word coi là toán TRONG DÒNG, nên phân số và tổng bị
    thu nhỏ. Số hiệu theo cách của chính Word: mảng phương trình có '#(n)' ở cuối."""
    body = _P(_lex(latex.strip())).seq()
    if number is not None:
        body = (f"<m:eqArr><m:e>{body}{_r('#', 'p')}{_r(f'({number})', 'p')}</m:e></m:eqArr>")
    # cỡ chữ ghi thẳng vào từng run toán (10 pt như thân bài), không phụ thuộc style
    # của đoạn chứa nó (ô bảng mặc định Normal 12 pt của template)
    sz = f'<w:rPr><w:sz w:val="{2 * size_pt}"/><w:szCs w:val="{2 * size_pt}"/></w:rPr>' if size_pt else ""
    if sz:
        body = re.sub(r"(<m:r>(?:<m:rPr>.*?</m:rPr>)?)", lambda m: m.group(1) + sz, body)
    return (f'<m:oMathPara xmlns:m="{M_NS}" xmlns:w="{W_NS}"><m:oMathParaPr><m:jc m:val="center"/>'
            f"</m:oMathParaPr><m:oMath>{body}</m:oMath></m:oMathPara>")


# Ước lượng bề rộng phương trình (đơn vị: "em" của Cambria Math) để tự ngắt dòng.
# Vì sao cần: cột IEEE 3,33 in, Word KHÔNG tự ngắt phương trình trưng bày nằm trong ô
# bảng, nên phương trình (3) và (5) của bài cost-aware tràn sang cột bên.
_REL = {"=", "ge", "le", "geq", "leq", "approx"}


def _w_seq(toks, k, stop=None):
    w = 0.0
    while k < len(toks):
        typ, val = toks[k]
        if stop and typ == stop:
            return w, k
        dw, k = _w_atom(toks, k)
        while k < len(toks) and toks[k][0] in "_^":
            sw, k = _w_arg(toks, k + 1)
            dw += 0.9 * sw
        w += dw
    return w, k


def _w_arg(toks, k):
    if k < len(toks) and toks[k][0] == "{":
        w, k = _w_seq(toks, k + 1, stop="}")
        return w, k + 1
    return _w_atom(toks, k)


def _w_atom(toks, k):
    typ, val = toks[k]
    if typ == "{":
        w, k = _w_seq(toks, k + 1, stop="}")
        return w, k + 1
    if typ == "ch":
        if val in "=+−-<>":
            return 1.1, k + 1               # dấu nhị phân kèm khoảng trắng hai bên
        if val in "()[]|,.":
            return 0.35, k + 1
        return 0.55, k + 1
    if typ == "cmd":
        if val in _SPACES:
            return {"quad": 1.0, "qquad": 2.0, ",": 0.17, ";": 0.28, " ": 0.25}.get(val, 0.0), k + 1
        if val in _SIZERS:
            return 0.0, k + 1
        if val == "frac":
            a, k = _w_arg(toks, k + 1)
            b, k = _w_arg(toks, k)
            return max(a, b) + 0.3, k
        if val == "sqrt":
            a, k = _w_arg(toks, k + 1)
            return a + 0.8, k
        if val in _ACCENTS:
            return _w_arg(toks, k + 1)
        if val in ("mathrm", "text", "operatorname", "textrm", "mathbb"):
            a, k2 = 0.0, k + 1
            if toks[k2][0] == "{":
                k2 += 1
                while toks[k2][0] != "}":
                    a += 0.5
                    k2 += 1
                return a, k2 + 1
            return 0.6, k2 + 1
        if val in _FUNCS:
            return 0.5 * len(_FUNCS[val]), k + 1
        if val in ("{", "}"):
            return 0.45, k + 1
        if val in _OPS or val in _REL:
            return 1.1, k + 1
        return 0.7, k + 1
    return 0.5, k + 1


def est_width_em(latex):
    toks = _lex(latex)
    w, _ = _w_seq(toks, 0)
    return w


def _top_splits(toks):
    """Vị trí token ở mức ngoài cùng (ngoài {} và ngoài ( ) [ ]) có thể ngắt dòng
    TRƯỚC nó: quan hệ (=, ≥…) và phép +, −. Trả [(k, loại)]."""
    out, brace, paren = [], 0, 0
    for k, (typ, val) in enumerate(toks):
        if typ == "{":
            brace += 1
        elif typ == "}":
            brace -= 1
        elif typ == "ch" and val in "([":
            paren += 1
        elif typ == "ch" and val in ")]":
            paren -= 1
        if brace or paren or k == 0:
            continue
        if (typ == "ch" and val == "=") or (typ == "cmd" and val in _REL):
            out.append((k, "rel"))
        elif typ == "ch" and val in "+-":
            out.append((k, "bin"))
    return out


def _detok(toks):
    s = []
    for typ, val in toks:
        if typ == "cmd":
            s.append("\\" + val + (" " if val[:1].isalpha() else ""))
        else:
            s.append(val)
    return "".join(s)


def break_long(latex, max_em):
    """Ngắt một phương trình quá rộng thành nhiều dòng tại mức ngoài cùng: ưu tiên
    trước dấu quan hệ thứ hai (hoặc sau dấu '=' đầu nếu vế trái ngắn), rồi trước dấu
    + gần giữa nhất. Không đổi ký hiệu nào; chỉ chọn chỗ xuống dòng."""
    if est_width_em(latex) <= max_em:
        return [latex]
    toks = _lex(latex)
    cands = _top_splits(toks)
    if not cands:
        return [latex]
    best = None
    for k, kind in cands:
        a, b = _detok(toks[:k]), _detok(toks[k:])
        wa, wb = est_width_em(a), est_width_em(b)
        if kind == "rel" and k == cands[0][0] and toks[k][1] == "=":
            # ngắt ngay sau '=' đầu tiên: vế trái giữ dấu '='
            a, b = _detok(toks[:k + 1]), _detok(toks[k + 1:])
            wa, wb = est_width_em(a), est_width_em(b)
        score = max(wa, wb) + (0 if kind == "rel" else 0.5)
        if best is None or score < best[0]:
            best = (score, a, b)
    _, a, b = best
    return break_long(a, max_em) + break_long(b, max_em)
