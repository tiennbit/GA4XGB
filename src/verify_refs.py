# -*- coding: utf-8 -*-
"""Xác minh tài liệu tham khảo qua Crossref API — chống bịa trích dẫn.

Dùng 2 chế độ:
  1) Tra theo tiêu đề:  python3 src/verify_refs.py --title "XGBoost: A Scalable Tree Boosting System"
  2) Tra theo DOI:      python3 src/verify_refs.py --doi 10.1109/ACCESS.2025.3596520
  3) Lô từ file:        python3 src/verify_refs.py --batch candidates.txt   (mỗi dòng 1 tiêu đề)

In ra metadata thẩm quyền: tác giả, tạp chí, năm, tập, trang, DOI.
Nếu Crossref không tìm thấy hoặc tiêu đề lệch -> báo NOT_FOUND / TITLE_MISMATCH -> LOẠI.
"""
import argparse
import os
import json
import sys
import time
import urllib.parse
import urllib.request

# Crossref "polite pool": đặt email của bạn để được ưu tiên hàng đợi.
#   export CROSSREF_MAILTO="ban@vidu.edu.vn"
# Không hard-code email cá nhân ở đây — repo này công khai.
MAILTO = os.environ.get("CROSSREF_MAILTO", "")
UA = (f"GA4XGB-refcheck/1.0 (mailto:{MAILTO})" if MAILTO
      else "GA4XGB-refcheck/1.0")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "into"}


def norm(s):
    return "".join(c.lower() for c in s if c.isalnum())


def tokens(s):
    """Tập từ có nghĩa (bỏ stopword) — dùng cho so khớp tiêu đề."""
    words = "".join(c.lower() if c.isalnum() else " " for c in s).split()
    return {w for w in words if w not in STOP and len(w) > 1}


def fmt(it):
    authors = it.get("author", [])
    names = "; ".join(
        f"{a.get('family','?')}, {''.join(p[0]+'.' for p in a.get('given','').split() if p)}"
        for a in authors[:8]
    )
    if len(authors) > 8:
        names += "; et al."
    title = (it.get("title") or ["?"])[0]
    venue = (it.get("container-title") or ["?"])[0]
    year = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
    vol = it.get("volume", "")
    issue = it.get("issue", "")
    page = it.get("page", "")
    doi = it.get("DOI", "")
    typ = it.get("type", "")
    return {"authors": names, "title": title, "venue": venue, "year": year,
            "volume": vol, "issue": issue, "pages": page, "doi": doi, "type": typ}


def by_doi(doi):
    try:
        d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        return fmt(d["message"])
    except Exception as e:
        return {"error": f"NOT_FOUND ({e})", "doi": doi}


def by_title(title, threshold=0.72):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3,
                                "mailto": MAILTO})
    try:
        d = _get(f"https://api.crossref.org/works?{q}")
    except Exception as e:
        return {"error": f"QUERY_FAILED ({e})", "title": title}
    items = d["message"].get("items", [])
    if not items:
        return {"error": "NOT_FOUND", "title": title}
    best = fmt(items[0])
    # Khớp tiêu đề: chuỗi con HOẶC Jaccard ĐỐI XỨNG |a∩b|/|a∪b|.
    # Phải đối xứng: bất đối xứng bỏ sót trường hợp kết quả có từ lạ
    # (vd. "Delving into Deep Imbalanced Regression" vs "Deep Knnor for
    #  Imbalanced Regression" — 3/4 từ trùng nhưng là 2 bài khác nhau).
    a, b = norm(title), norm(best["title"])
    if not (a in b or b in a):
        ta, tb = tokens(title), tokens(best["title"])
        jac = len(ta & tb) / max(len(ta | tb), 1)
        if jac < threshold:
            diff = (ta ^ tb)
            best["warning"] = (f"TITLE_MISMATCH: Jaccard {jac:.0%} < {threshold:.0%}; "
                               f"từ lệch: {sorted(diff)[:6]}. Crossref có thể KHÔNG "
                               f"index bài này (vd. PMLR/arXiv) -> tra tay. "
                               f"Query: '{title}'")
    return best


def show(r):
    if "error" in r:
        print(f"  ✗ {r['error']}  <- {r.get('title', r.get('doi',''))}")
        return False
    w = r.get("warning", "")
    mark = "⚠" if w else "✓"
    print(f"  {mark} {r['authors']}. \"{r['title']}\". {r['venue']}, "
          f"vol. {r['volume']}, no. {r['issue']}, pp. {r['pages']}, {r['year']}. "
          f"DOI: {r['doi']}  [{r['type']}]")
    if w:
        print(f"     {w}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title")
    ap.add_argument("--doi")
    ap.add_argument("--batch", help="file: mỗi dòng 1 tiêu đề (bỏ qua dòng trống/#)")
    args = ap.parse_args()

    if args.doi:
        show(by_doi(args.doi))
    elif args.title:
        show(by_title(args.title))
    elif args.batch:
        lines = [l.strip() for l in open(args.batch) if l.strip() and not l.startswith("#")]
        ok = 0
        for i, t in enumerate(lines, 1):
            print(f"[{i}/{len(lines)}] {t[:70]}")
            ok += show(by_title(t))
            time.sleep(0.4)   # lịch sự với API
        print(f"\n{ok}/{len(lines)} xác minh được.")
    else:
        ap.print_help()
