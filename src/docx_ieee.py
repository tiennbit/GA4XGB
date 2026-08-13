# -*- coding: utf-8 -*-
"""Nguyên thuỷ OOXML cho template IEEE Access.

Tách riêng khỏi build_docx.py vì đây là phần "luật của tạp chí", còn build_docx.py
là phần "nội dung bài này". Ba luật quan trọng rút ra từ việc soi template:

1. TEMPLATE CÓ 2 LOẠI SECTION. sec0 = 1 cột (tiêu đề + tác giả + abstract chạy
   hết chiều ngang trang), sec1..8 = 2 cột (thân bài). Bản dựng cũ chỉ có MỘT
   section 2 cột -> tiêu đề bài bị nhồi vào cột trái. Đó là lỗi format nặng nhất.

2. CÁC STYLE TIÊU ĐỀ TỰ ĐỘNG ĐÁNH SỐ. H1_List (numId 4 -> "I."), H2_First/H2_Cont
   (upperLetter), H3 (numId 6 -> "1)"), và cả Fig Caption (numId 5 -> "FIGURE n.").
   Nếu ta gõ số vào text thì Word in RA HAI LẦN: "FIGURE 1. FIGURE 1. ...".
   Ta chọn gõ số thủ công (kiểm soát được thứ tự, không phụ thuộc Word) nên phải
   TẮT numbering ở cấp đoạn bằng numId = 0.

3. BẢNG THEO LUẬT IEEE: không có kẻ dọc; viền ngoài trên/dưới sz=12 màu 808080;
   hàng tiêu đề viền trên đôi sz=6 và viền dưới đơn sz=6; thân bảng không viền.
   Bản dựng cũ dùng style "Normal Table" -> không có đường kẻ nào.
"""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Bề rộng khả dụng: khổ 8in - lề 0.514×2 = 6.97in. Một cột = (6.97 - 0.28)/2.
FULL_W = Inches(6.95)
COL_W = Inches(3.33)


def clear_body(doc):
    """Xoá nội dung mẫu, GIỮ sectPr cuối (nó mang khổ giấy/lề/header)."""
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)


def _cols(section):
    sectPr = section._sectPr
    c = sectPr.find(qn("w:cols"))
    if c is None:
        c = OxmlElement("w:cols")
        sectPr.append(c)
    return c


def set_columns(section, num, space=400):
    c = _cols(section)
    c.set(qn("w:num"), str(num))
    c.set(qn("w:space"), str(space))
    c.set(qn("w:equalWidth"), "1")


def no_autonum(par):
    """Tắt đánh số tự động cho ĐOẠN này (numId=0), giữ nguyên font/khoảng cách
    của style. Phải chèn numPr ngay sau pStyle vì schema OOXML quy định thứ tự."""
    pPr = par._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:numPr")):
        pPr.remove(old)
    numPr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    nid = OxmlElement("w:numId")
    nid.set(qn("w:val"), "0")
    numPr.append(ilvl)
    numPr.append(nid)
    style = pPr.find(qn("w:pStyle"))
    if style is not None:
        style.addnext(numPr)
    else:
        pPr.insert(0, numPr)
    return par


def set_indent(par, left=0, first=0, hanging=None):
    pPr = par._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:ind")):
        pPr.remove(old)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), str(left))
    if hanging is not None:
        ind.set(qn("w:hanging"), str(hanging))
    else:
        ind.set(qn("w:firstLine"), str(first))
    pPr.append(ind)
    return par


def _border(el, tag, val, sz, color="000000"):
    b = OxmlElement(f"w:{tag}")
    b.set(qn("w:val"), val)
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "0")
    b.set(qn("w:color"), color)
    el.append(b)


def format_table_ieee(table, widths_in, font_pt=7.0):
    """Áp luật kẻ bảng IEEE + khoá bề rộng cột (tblLayout fixed mới có tác dụng).

    widths_in: list bề rộng từng cột theo inch; tổng phải bằng bề rộng bảng.
    """
    tbl = table._tbl
    tblPr = tbl.tblPr

    for tag in ("tblBorders", "tblStyle", "tblW", "tblLayout"):
        for old in tblPr.findall(qn(f"w:{tag}")):
            tblPr.remove(old)

    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(sum(widths_in) * 1440)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)

    bd = OxmlElement("w:tblBorders")
    _border(bd, "top", "single", 12, "808080")
    _border(bd, "bottom", "single", 12, "808080")
    tblPr.append(bd)

    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)

    table.autofit = False
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            cell.width = Inches(widths_in[j])
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            if i == 0:                       # hàng tiêu đề: trên đôi, dưới đơn
                tb = OxmlElement("w:tcBorders")
                _border(tb, "top", "double", 6)
                _border(tb, "bottom", "single", 6)
                tcPr.append(tb)
            for p in cell.paragraphs:
                pf = p.paragraph_format
                pf.space_before = Pt(1)
                pf.space_after = Pt(1)
                pf.line_spacing = 1.0
                for r in p.runs:
                    r.font.size = Pt(font_pt)
                    r.font.name = "Times New Roman"
    # khoá bề rộng cột ở cấp <w:gridCol> nữa, nếu không Word tự co lại
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, wi in zip(grid.findall(qn("w:gridCol")), widths_in):
            gc.set(qn("w:w"), str(int(wi * 1440)))


def repeat_header(table):
    """Đánh dấu hàng 1 lặp lại khi bảng tràn sang cột/trang sau."""
    trPr = table.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trPr.append(h)


def shrink_break(doc):
    """Đoạn mang sect-break (python-docx tự chèn) mặc định style Normal 12pt ->
    Word in ra một dòng trống thấy rõ ở chỗ đổi số cột. Ép về 1pt để nó tàng hình."""
    p = doc.paragraphs[-1]
    pPr = p._p.get_or_add_pPr()
    if pPr.find(qn("w:sectPr")) is None:
        return
    rPr = OxmlElement("w:rPr")
    for tag in ("w:sz", "w:szCs"):
        e = OxmlElement(tag)
        e.set(qn("w:val"), "2")          # 2 nửa-point = 1pt
        rPr.append(e)
    pPr.insert(0, rPr)
    sp = OxmlElement("w:spacing")
    for k in ("w:before", "w:after", "w:line"):
        sp.set(qn(k), "0")
    sp.set(qn("w:lineRule"), "auto")
    pPr.append(sp)


# --------------------------------------------------------------- trích dẫn
# Trong bản PDF của IEEE Access, "[4]" ở thân bài bấm được và nhảy xuống mục
# REFERENCES. Muốn có điều đó thì mỗi mục tài liệu phải mang một BOOKMARK, và
# mỗi "[4]" trong thân bài phải là <w:hyperlink w:anchor="...">.
# Cố ý KHÔNG dùng style "Hyperlink": IEEE in trích dẫn màu đen như chữ thường,
# xanh-gạch-chân sẽ sai format. Liên kết vẫn bấm được, chỉ là không phô ra.

def ref_anchor(n):
    return f"ref{int(n)}"


def add_bookmark(par, name, bid):
    """Bọc TOÀN BỘ đoạn trong một bookmark (start ở đầu, end ở cuối)."""
    p = par._p
    st = OxmlElement("w:bookmarkStart")
    st.set(qn("w:id"), str(bid))
    st.set(qn("w:name"), name)
    en = OxmlElement("w:bookmarkEnd")
    en.set(qn("w:id"), str(bid))
    pPr = p.find(qn("w:pPr"))
    if pPr is not None:
        pPr.addnext(st)
    else:
        p.insert(0, st)
    p.append(en)


def add_cite_link(par, text, anchor):
    """Chèn một run là liên kết nội bộ tới anchor, giữ nguyên hình thức chữ."""
    h = OxmlElement("w:hyperlink")
    h.set(qn("w:anchor"), anchor)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    par._p.append(h)
